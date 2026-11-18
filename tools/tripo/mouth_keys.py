"""The jaw-open and mouth-close shape keys of Human as offsets per vertex (mouth_arap.py tells the lips apart by them).  blender -b X.blend --python mouth_keys.py -- OUT.npz"""
import bpy, numpy as np, sys
out = sys.argv[sys.argv.index('--') + 1]
o = bpy.data.objects['Human']; kb = o.data.shape_keys.key_blocks
n = len(o.data.vertices)
def co(k):
    a = np.zeros(n * 3); kb[k].data.foreach_get('co', a); return a.reshape(-1, 3)
b = co(kb[0].name)
np.savez(out, basis=b, **{k_: co(k_) - b for k_ in ('jawOpen', 'mouthClose', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight')})
print('DUMP', n)
