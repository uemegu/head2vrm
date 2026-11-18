"""blender -b -P dump_verts.py -- GLB OUT.npz : world vertex positions and triangles of all meshes"""
import bpy, sys, numpy as np
glb, out = sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=glb)
V=[];F=[];n=0
for o in bpy.context.scene.objects:
    if o.type!='MESH': continue
    m=o.data; m.calc_loop_triangles()
    V+= [tuple(o.matrix_world@v.co) for v in m.vertices]; F+=[[i+n for i in t.vertices] for t in m.loop_triangles]; n+=len(m.vertices)
np.savez(out,V=np.array(V),F=np.array(F))
