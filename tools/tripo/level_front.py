import bpy, sys, mathutils, numpy as np
glb, out = sys.argv[sys.argv.index('--') + 1:]
dist = float(__import__('os').environ.get('DIST', 0))   # 0: orthographic; else a perspective camera this far (in head units)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
sc = bpy.context.scene
obs = [o for o in sc.objects if o.type == 'MESH']
for o in obs:
    for m in o.data.materials:
        nt = m.node_tree; b = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'); outn = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
        em = nt.nodes.new('ShaderNodeEmission'); nt.links.new(b.inputs['Base Color'].links[0].from_socket, em.inputs[0]); nt.links.new(em.outputs[0], outn.inputs['Surface'])
V = np.array([o.matrix_world @ v.co for o in obs for v in o.data.vertices])
lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2; size = (hi - lo).max()
sc.render.engine = 'BLENDER_EEVEE'; sc.view_settings.view_transform = 'Standard'
sc.render.resolution_x = sc.render.resolution_y = 2000
w = bpy.data.worlds.new('w'); sc.world = w; w.color = (0.02, 0.02, 0.02)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam
# glTF import: Blender z up, front is -y
if dist:
    cam.data.type = 'PERSP'; cam.data.sensor_width = 36; d = dist * size
    cam.data.lens = 36 * d / (size * 1.1)
    cam.location = mathutils.Vector((c[0], c[1] - d, c[2]))
else:
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = size * 1.1; cam.location = mathutils.Vector((c[0], c[1] - 5 * size, c[2]))
cam.rotation_euler = (np.pi / 2, 0, 0); cam.data.clip_end = 100 * size
sc.render.filepath = out; bpy.ops.render.render(write_still=True); print('FRONT', V.shape, size)
