"""Which skin texels sit under the grafted Tripo hair: a ray out along the skin normal meets the hair within REACH mm.
The face Tripo gave (within NEAR mm of its surface and facing forward) is kept out.
blender --background textured_hair.blend --python scalp_mask.py -- OUT_DIR MASK.npy   (mask at 1/STEP of the texel grid)"""
import os, sys
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

O, out = sys.argv[sys.argv.index('--') + 1:]
STEP, REACH, NEAR = 4, float(os.environ.get('REACH', 40)) / 1000, float(os.environ.get('NEAR', 3)) / 1000
hair = bpy.data.objects['tripo_hair']; human = bpy.data.objects['Human']
dg = bpy.context.evaluated_depsgraph_get()
bvh = BVHTree.FromObject(hair, dg)            # in hair-local space
toh = hair.matrix_world.inverted() @ human.matrix_world
t = np.load(f'{O}/texels.npz'); pos, nrm, have = t['pos'][::STEP, ::STEP], t['nrm'][::STEP, ::STEP], t['have'][::STEP, ::STEP]
dq = np.load(f"{O}/tripo_dist_q.npy")
mask = np.zeros(have.shape, np.float32)
# heights from the eyes, in one reference head's units (eyes 1.440, top 1.554) scaled by this head's eye-to-top
eyes = bpy.data.objects['Human.high-poly']
eye_z = float(np.mean([(eyes.matrix_world @ v.co).z for v in eyes.data.vertices]))
top_z = max((human.matrix_world @ v.co).z for v in human.data.vertices)
k = (top_z - eye_z) / 0.114
at = lambda dz: eye_z + dz * k
ys, xs = np.nonzero(have & (pos[..., 2] > at(-0.19)))
d = dq[ys, xs]
keep_face = (d < NEAR) & ((nrm[ys, xs, 1] < 0.2) | (np.abs(pos[ys, xs, 0]) > 0.062 * k))   # Tripo's face and ears, not the back of its neck
CROWN = at(float(os.environ.get('CROWN', 0.02)))   # above this every non-face texel is scalp (the hair part has holes on the crown)
BACK_Z, BACK_Y = at(float(os.environ.get('BACK_Z', -0.04))), float(os.environ.get('BACK_Y', 0.02)) * k   # and behind the ears above the nape (holes at the back)
print('SCALP_MASK eyes %.3f top %.3f scale %.2f' % (eye_z, top_z, k))
R3 = np.array(toh.to_3x3())
P = pos[ys, xs] @ np.array(toh)[:3, :3].T + np.array(toh)[:3, 3]
N = nrm[ys, xs] @ R3.T; N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
hit = 0
for i in range(len(ys)):
    if keep_face[i]: continue
    p_ = pos[ys[i], xs[i]]
    if p_[2] > CROWN or (p_[2] > BACK_Z and p_[1] > BACK_Y):
        mask[ys[i], xs[i]] = 1; hit += 1; continue
    o = Vector(P[i] + N[i] * 0.0005)
    loc, _, _, dist = bvh.ray_cast(o, Vector(N[i]), REACH)
    if loc is not None:
        mask[ys[i], xs[i]] = 1; hit += 1
np.save(out, mask)
print('SCALP_MASK head texels', len(ys), 'face kept', int(keep_face.sum()), 'under hair', hit)
