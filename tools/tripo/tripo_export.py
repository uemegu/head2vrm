"""Input head GLB -> mesh npz (world verts, tris, uv per corner, texture) + flat-lit orthographic renders for landmarks.

blender --background --python tripo_export.py -- HEAD_GLB OUT_DIR
Renders use the texture as emission (no lighting), Standard view transform.
"""
import json, math, os, sys
import bpy, mathutils
import numpy as np

glb, out = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
obj = [o for o in bpy.data.objects if o.type == 'MESH'][0]
me = obj.data
mw = np.array(obj.matrix_world)
n = len(me.vertices)
co = np.zeros(n * 3); me.vertices.foreach_get('co', co); co = co.reshape(n, 3) @ mw[:3, :3].T + mw[:3, 3]
me.calc_loop_triangles()
tris = np.array([t.vertices[:] for t in me.loop_triangles])
uvl = np.zeros(len(me.loops) * 2); me.uv_layers.active.data.foreach_get('uv', uvl); uvl = uvl.reshape(-1, 2)
tuv = np.array([[uvl[l] for l in t.loops] for t in me.loop_triangles])
img = [i for i in bpy.data.images if i.size[0] > 0][0]
img.filepath_raw = os.path.join(out, 'tripo_basecolor.png'); img.file_format = 'PNG'; img.save()
np.savez(os.path.join(out, 'tripo_mesh.npz'), co=co, tris=tris, tuv=tuv)

mat = me.materials[0]
nt = mat.node_tree
tex = [nd for nd in nt.nodes if nd.type == 'TEX_IMAGE'][0]
em = nt.nodes.new('ShaderNodeEmission'); outn = [nd for nd in nt.nodes if nd.type == 'OUTPUT_MATERIAL'][0]
nt.links.new(tex.outputs['Color'], em.inputs['Color']); nt.links.new(em.outputs[0], outn.inputs['Surface'])
sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'; sc.view_settings.view_transform = 'Standard'
sc.render.resolution_x = sc.render.resolution_y = 1024
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True; w.node_tree.nodes['Background'].inputs[0].default_value = (0.2, 0.2, 0.2, 1)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = 'ORTHO'
lo, hi = co.min(0), co.max(0)
head = co[co[:, 2] > lo[2] + 0.15 * (hi[2] - lo[2])]   # above the shoulders
ctr = (head.min(0) + head.max(0)) / 2
size = float((head.max(0) - head.min(0))[[0, 2]].max()) * 1.05
cam.data.ortho_scale = size
cams = []
for ang in (0, -35, 35):
    a = math.radians(ang)
    loc = mathutils.Vector((ctr[0] + math.sin(a) * 3, ctr[1] - math.cos(a) * 3, ctr[2]))
    cam.location = loc
    cam.rotation_euler = (mathutils.Vector(ctr) - loc).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()
    path = os.path.join(out, f'view_{ang:+d}.png')
    sc.render.filepath = path; bpy.ops.render.render(write_still=True)
    cams.append({'angle': ang, 'matrix_world': [list(r) for r in cam.matrix_world], 'ortho_scale': size, 'size': 1024, 'path': path})
json.dump(cams, open(os.path.join(out, 'views.json'), 'w'), indent=1)
print('EXPORTED', n, len(tris), img.size[:])
