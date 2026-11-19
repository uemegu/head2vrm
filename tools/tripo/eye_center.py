"""Each eyeball moved sideways (x) to the middle of its opening, so the eyes look straight ahead. Any face, after the lid margins are set.
blender -b warped.blend -P eye_center.py -- OUT_DIR OUT_BLEND      (OUT_DIR: run.sh's, with lm3d.npy and warp.npz)
tripo_fit.py moves the eyeballs with the lids, but seam_fit.py then moves only the margins onto the painted lines: on one test head the
irises sat 2 % of the eye's width outside the openings' middles (both outward, 0.48 / 0.52 between the corners).
The opening: the middle of the input head's own corner landmarks (MediaPipe 33/133, 362/263 on its head, lm3d.npy, in MPFB's frame by
the fit's similarity, warp.npz); seam_fit.py puts the lid corners on them. The mesh margin's x extremes were not the corners one
sees (they moved the eyes outward). Checked on the viewer: gaze_check.py (the eyeball's +z pole between MediaPipe's corners).
The eyeball (Human.high-poly, every shape key too) moves by middle - centre in x."""
import sys
import bpy
import numpy as np
O, out = sys.argv[sys.argv.index('--') + 1:]
P = np.load(f'{O}/lm3d.npy'); w = np.load(f'{O}/warp.npz')
Q = (w['scale'] * (w['rot'] @ P.T)).T + w['trans']
hum, eye = bpy.data.objects['Human'], bpy.data.objects['Human.high-poly']
Mh, Me = np.array(hum.matrix_world), np.array(eye.matrix_world)
get = lambda d, n: (lambda a: (d.foreach_get('co', a), a.reshape(n, 3))[1])(np.zeros(n * 3))
nh, ne = len(hum.data.vertices), len(eye.data.vertices)
V = get(hum.data.vertices, nh) @ Mh[:3, :3].T + Mh[:3, 3]
N = np.zeros(nh * 3); hum.data.vertices.foreach_get('normal', N); N = N.reshape(nh, 3) @ Mh[:3, :3].T
E = get(eye.data.vertices, ne) @ Me[:3, :3].T + Me[:3, 3]
ed = np.zeros(len(hum.data.edges) * 2, int); hum.data.edges.foreach_get('vertices', ed); ed = ed.reshape(-1, 2)
dx_side = {}
for sgn in (1, -1):
    e = E[np.sign(E[:, 0]) == sgn]; c = (e.min(0) + e.max(0)) / 2; r = (e.max(0) - e.min(0)).max() / 2
    a, b = (263, 362) if sgn > 0 else (33, 133)
    if np.sign(Q[a, 0]) != sgn:
        a, b = (33, 133) if sgn > 0 else (263, 362)
    mid = (Q[a, 0] + Q[b, 0]) / 2
    dx_side[sgn] = mid - c[0]
    print('EYE_CENTER side %+d: corners %.1f..%.1f mm, ball centre %.1f, move %.2f mm' % (sgn, Q[a, 0] * 1000, Q[b, 0] * 1000, c[0] * 1000, dx_side[sgn] * 1000))
inv = np.linalg.inv(Me[:3, :3])
base = get(eye.data.vertices, ne)
side = np.sign(E[:, 0])
off_w = np.zeros((ne, 3)); off_w[:, 0] = np.where(side > 0, dx_side[1], dx_side[-1])
off = off_w @ inv.T
eye.data.vertices.foreach_set('co', (base + off).ravel())
for k in (eye.data.shape_keys.key_blocks if eye.data.shape_keys else []):
    k.data.foreach_set('co', (get(k.data, ne) + off).ravel())
eye.data.update()
bpy.ops.wm.save_as_mainfile(filepath=out)
