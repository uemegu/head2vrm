"""Export basemesh body triangles with UVs, world positions and normals -> NPZ.

blender --background warped.blend --python export_uv.py -- OUT_NPZ
"""
import sys
import bpy
import numpy as np

out = sys.argv[sys.argv.index('--') + 1]
body = bpy.data.objects['Human']
me = body.data
mw = np.array(body.matrix_world)
gi = body.vertex_groups['body'].index
in_body = np.array([any(g.group == gi and g.weight > 0.5 for g in v.groups) for v in me.vertices])
co = np.array([v.co for v in me.vertices]) @ mw[:3, :3].T + mw[:3, 3]
nrm = np.array([v.normal for v in me.vertices]) @ np.linalg.inv(mw[:3, :3])
uv = me.uv_layers.active.data
tv, tuv = [], []
for p in me.polygons:
    vs = list(p.vertices)
    if not in_body[vs].all():
        continue
    ls = list(p.loop_indices)
    for k in range(1, len(vs) - 1):
        tv.append((vs[0], vs[k], vs[k + 1]))
        tuv.append([uv[ls[0]].uv[:], uv[ls[k]].uv[:], uv[ls[k + 1]].uv[:]])
eye = next(o for o in bpy.data.objects if o.type == 'MESH' and 'high-poly' in o.name)
emw = np.array(eye.matrix_world)
eye_co = np.array([v.co for v in eye.data.vertices]) @ emw[:3, :3].T + emw[:3, 3]
np.savez(out, co=co, nrm=nrm, tv=np.array(tv), tuv=np.array(tuv), eye_co=eye_co)
print('EXPORTED', len(tv))
