"""Slim the lower face of a level Tripo head: each vertex in front of the ears moves toward the midline by s(z) (slim_target.py,
both sides' mean so the face stays symmetric), scaled by how far out it is (0 inside INNER of the half-width: nose, mouth and
eyes untouched) and by depth (full in front of the jaw angle, 0 at the ear). UVs and texture unchanged.
blender -b -P slim_apply.py -- HEAD.glb SLIM.npz OUT.glb"""
import bpy, sys, os, numpy as np
glb, sf, out = sys.argv[sys.argv.index('--')+1:]
S = np.load(sf); z = S['z']; s = (S['s_L'] + S['s_R']) / 2; w = np.nanmean([S['w_L'], S['w_R']], 0); mx = float(S['mid_x'])
INNER = float(os.environ.get('INNER', 0.45)); Y0 = float(os.environ.get('Y0', -0.03)); Y1 = float(os.environ.get('Y1', 0.04))
ss = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=glb)
moved = 0; mx_d = 0
for o in bpy.context.scene.objects:
    if o.type != 'MESH': continue
    M = o.matrix_world; Mi = M.inverted()
    for v in o.data.vertices:
        p = M @ v.co
        if p.z < z[0] or p.z > z[-1]: continue
        sv = np.interp(p.z, z, s); wv = np.interp(p.z, z, np.nan_to_num(w, nan=1))
        if sv <= 0: continue
        ax = abs(p.x - mx); a = ss((ax / wv - INNER) / (0.95 - INNER)); b = 1 - ss((p.y - Y0) / (Y1 - Y0))
        d = sv * a * b
        if d <= 0: continue
        p.x -= np.sign(p.x - mx) * d; v.co = Mi @ p; moved += 1; mx_d = max(mx_d, d)
print('SLIM_APPLY moved', moved, 'max %.4f' % mx_d)
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB')
