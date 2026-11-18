"""The eyeball mesh (Human.high-poly) in world space with its UVs -> npz (co, tris, tuv).   blender -b IN.blend -P eye_dump_cam.py -- OUT.npz"""
import sys
import bpy
import numpy as np
out = sys.argv[sys.argv.index('--') + 1]
o = bpy.data.objects['Human.high-poly']; me = o.data
mw = np.array(o.matrix_world)
co = np.array([v.co for v in me.vertices]) @ mw[:3, :3].T + mw[:3, 3]
me.calc_loop_triangles()
uv = me.uv_layers.active.data
tris = np.array([t.vertices[:] for t in me.loop_triangles])
tuv = np.array([[uv[l].uv[:] for l in t.loops] for t in me.loop_triangles])
np.savez(out, co=co, tris=tris, tuv=tuv)
print('EYE_DUMP', len(co), len(tris))
