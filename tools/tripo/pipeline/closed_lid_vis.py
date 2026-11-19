"""Which skin texels show only when the eyes are closed: the face rendered with its UVs as colour (Cycles, 1 sample, no
filter) from a few cameras round the eyes, open and with both blink keys at 1.
blender -b face_only.blend -P closed_lid_vis.py -- OUT_DIR [KEY_L KEY_R]   -> OUT_DIR/uv_{open,closed}_{cam}.exr

Why: the painted lashes are kept on the lid (LASHES=painted); closing the eye stretches the lid skin over the eye, and the
painted strokes there show.
"""
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
out = argv[0]; kl, kr = (argv[1], argv[2]) if len(argv) > 2 else ('eyeBlinkLeft', 'eyeBlinkRight')
sc = bpy.context.scene
for o in sc.objects:
    o.hide_render = o.name not in ('Human', 'Human.high-poly')
H = bpy.data.objects['Human']
eye = np.array([list(v.co) for v in bpy.data.objects['Human.high-poly'].data.vertices])
mw = bpy.data.objects['Human.high-poly'].matrix_world
eye = np.array([list(mw @ Vector(p)) for p in eye]); c = eye.mean(0)
print('CLOSED_VIS eyes centre', c.round(4))
# materials: the skin's UVs as colour (b = 1 marks skin), the eyeballs black
m = bpy.data.materials.new('uvvis'); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
uv = nt.nodes.new('ShaderNodeUVMap'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); comb = nt.nodes.new('ShaderNodeCombineXYZ')
em = nt.nodes.new('ShaderNodeEmission'); o_ = nt.nodes.new('ShaderNodeOutputMaterial')
nt.links.new(uv.outputs['UV'], sep.inputs[0]); nt.links.new(sep.outputs[0], comb.inputs[0]); nt.links.new(sep.outputs[1], comb.inputs[1])
comb.inputs[2].default_value = 1.0; nt.links.new(comb.outputs[0], em.inputs['Color']); nt.links.new(em.outputs[0], o_.inputs['Surface'])
mb = bpy.data.materials.new('black'); mb.use_nodes = True; mb.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0, 0, 0, 1)
for ob, mat in ((H, m), (bpy.data.objects['Human.high-poly'], mb)):
    ob.data.materials.clear(); ob.data.materials.append(mat)
    for p in ob.data.polygons: p.material_index = 0
sc.render.engine = 'CYCLES'; sc.cycles.samples = 1; sc.cycles.use_denoising = False; sc.cycles.pixel_filter_type = 'BOX'; sc.cycles.filter_width = 0.01
sc.view_settings.view_transform = 'Standard'; sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.use_nodes = False; sc.world.color = (0, 0, 0)
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 2400, 1200, 100
sc.render.image_settings.file_format = 'OPEN_EXR'; sc.render.image_settings.color_depth = '32'
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 200; cam.data.sensor_width = 36
keys = H.data.shape_keys.key_blocks
for k in keys: k.value = 0
# cameras: front, from above and below by 20 deg, from either side by 25 deg (0.5 m off the eyes' centre)
cams = {'front': (0, 0), 'up': (20, 0), 'down': (-20, 0), 'left': (0, 25), 'right': (0, -25)}
for state in ('open', 'closed'):
    keys[kl].value = keys[kr].value = 1.0 if state == 'closed' else 0.0
    for name, (el, az) in cams.items():
        e, a = np.radians(el), np.radians(az)
        d = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
        cam.location = Vector(c + d * 0.5)
        cam.rotation_euler = (Vector(c) - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = f'{out}/uv_{state}_{name}.exr'
        bpy.ops.render.render(write_still=True)
print('CLOSED_VIS done')
