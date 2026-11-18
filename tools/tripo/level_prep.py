"""A Tripo head GLB -> OUT_DIR/head.glb, the whole head turned level (level_head.sh measures the angle).
blender --background --python level_prep.py -- GLB OUT_DIR        env LEVEL (degrees), LEVEL_C (x,z of the turn's centre)
A leaning reference bakes its lean into Tripo's head. The whole head turns about the point between the eyes, in the face's
plane (x-z); the UV goes with it. The largest mesh part is the head; other parts (hair, scraps) are left out."""
import math, os, sys
import bpy
glb, out = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
objs = {o.name: o for o in bpy.data.objects if o.type == 'MESH'}
a = math.radians(float(os.environ['LEVEL']))
cx, cz = (float(v) for v in os.environ['LEVEL_C'].split(','))
head = max(objs.values(), key=lambda o: len(o.data.vertices))
me = head.data
for v in me.vertices:
    x, z = v.co.x - cx, v.co.z - cz
    v.co.x, v.co.z = cx + x * math.cos(a) - z * math.sin(a), cz + x * math.sin(a) + z * math.cos(a)
me.update()
print('LEVEL %.2f deg about (%.4f, %.4f), %d verts' % (math.degrees(a), cx, cz, len(me.vertices)))
bpy.ops.object.select_all(action='DESELECT')
head.select_set(True)
bpy.context.view_layer.objects.active = head
bpy.ops.export_scene.gltf(filepath=os.path.join(out, 'head.glb'), use_selection=True, export_format='GLB', export_image_format='AUTO')
print('PREP head.glb')
