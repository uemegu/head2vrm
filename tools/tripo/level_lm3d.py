"""2D landmarks of an orthographic front render (level_front.py, 2000 px, ortho size*1.1) -> 3D points on the mesh by a ray from the front.
blender -b -P lm3d.py -- GLB LM.npy OUT.npy"""
import bpy, sys, numpy as np, mathutils
glb, lmf, out = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
V = np.array([o.matrix_world @ v.co for o in obs for v in o.data.vertices])
lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2; size = (hi - lo).max(); sc = size * 1.1
lm = np.load(lmf)[:, :2]
X = c[0] + (lm[:, 0] - 1000) / 2000 * sc
Z = c[2] - (lm[:, 1] - 1000) / 2000 * sc
dg = bpy.context.evaluated_depsgraph_get()
P = np.full((len(lm), 3), np.nan)
for i, (x, z) in enumerate(zip(X, Z)):
    hit, loc, *_ = bpy.context.scene.ray_cast(dg, mathutils.Vector((x, c[1] - 5 * size, z)), mathutils.Vector((0, 1, 0)))
    if hit: P[i] = loc
np.save(out, P); print('LM3D', np.isnan(P[:, 0]).sum(), 'misses')
