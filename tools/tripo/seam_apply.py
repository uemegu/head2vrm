"""Apply seam_fit.py's offsets to Human and its lashes, in every shape key. A blink key's lid is already carried to the lower lid, so there the
offset fades with the key's own travel. blender -b warped.blend -P seam_apply.py -- SEAM_NPZ OUT_BLEND"""
import sys

import bpy
import numpy as np

seam, out = sys.argv[sys.argv.index('--') + 1:]
d = np.load(seam)
hum = bpy.data.objects['Human']
inv = np.linalg.inv(np.array(hum.matrix_world)[:3, :3])


def get(h, n):
    x = np.zeros(n * 3); h.foreach_get('co', x); return x.reshape(n, 3)


def apply(me, off):
    n = len(me.vertices)
    base = get(me.vertices, n)
    me.vertices.foreach_set('co', (base + off).ravel())
    for k in (me.shape_keys.key_blocks if me.shape_keys else []):
        co = get(k.data, n)
        f = np.ones(n)
        if k.name.startswith('eyeBlink'):
            trav = np.linalg.norm(co - base, axis=1)
            f = 1 - np.clip(trav / max(trav.max(), 1e-6), 0, 1)
        k.data.foreach_set('co', (co + off * f[:, None]).ravel())
    me.update()


apply(hum.data, d['delta'] @ inv.T)
lash = bpy.data.objects.get('Human.eyelashes01')
if lash is None:   # a finished face without the base's lashes (LASHES=painted): Human only
    bpy.ops.wm.save_as_mainfile(filepath=out); raise SystemExit
L = np.array(lash.matrix_world)
n = len(lash.data.vertices)
p = get(lash.data.vertices, n) @ L[:3, :3].T + L[:3, 3]
off = np.zeros((n, 3))
best = np.full(n, 1e9)
for i in (1, 2):                                   # the eyes' margins; each lash vertex rides with its nearest margin point
    pos, val = d[f'pos{i}'], d[f'off{i}']
    dist = np.linalg.norm(p[:, None] - pos[None], axis=2)
    j = dist.argmin(1)
    m = dist[np.arange(n), j] < best
    off[m] = val[j[m]]
    best[m] = dist[np.arange(n), j][m]
apply(lash.data, off @ np.linalg.inv(L[:3, :3]).T)
print('SEAM_APPLY lash max mm', round(float(np.linalg.norm(off, axis=1).max()) * 1000, 2))
bpy.ops.wm.save_as_mainfile(filepath=out)
