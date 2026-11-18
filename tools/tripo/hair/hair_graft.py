"""Put a Tripo hair part (hairgraft/split.py) on the fitted MPFB character: the same similarity transform that
tripo_fit.py moved the Tripo head with (warp.npz scale/rot/trans), then skinned 100% to the head bone.
In front of the ears the hair is lifted off the skin where it touches or sinks into it.
The MPFB scalp (above the brows or behind the ears) is the base mesh's own skull and pokes through the hair in places: a
scalp vertex with no hair along its normal but hair within 12 mm behind it is pushed 4 mm under the hair
(a ring around it follows smoothly; every shape key gets the same offset).

blender --background textured_final.blend --python hair_graft.py -- HAIR_GLB WARP_NPZ OUT_BLEND
"""
import os
import sys
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

hair_glb, warp, out = sys.argv[sys.argv.index('--') + 1:]
LIFT, LIFT_REACH, LIFT_BELOW = 0.002, 0.006, 0.09
w = np.load(warp)
s, R, t = float(w['scale']), w['rot'], w['trans']
rig = bpy.data.objects['Human.rig']
human = bpy.data.objects['Human']
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=hair_glb)
new = [o for o in bpy.data.objects if o not in before]
hair = next(o for o in new if o.type == 'MESH')
me = hair.data
n = len(me.vertices)
co = np.zeros(n * 3); me.vertices.foreach_get('co', co)
mw = np.array(hair.matrix_world)
co = co.reshape(n, 3) @ mw[:3, :3].T + mw[:3, 3]           # world, as tripo_export.py read the head
co = (s * (R @ co.T)).T + t                                # -> basemesh frame (the Human mesh's local space)
if os.environ.get('BANG_NPZ'):                             # seam_fit.py: the bangs' inner edge on the portrait's
    co = co + np.load(os.environ['BANG_NPZ'])['offset']
hm = np.array(human.matrix_world)
co = co @ hm[:3, :3].T + hm[:3, 3]                         # -> world
for o in new:
    if o is not hair:
        bpy.data.objects.remove(o)
hair.parent = None
hair.matrix_world = rig.matrix_world.copy()
co = (co - np.array(rig.matrix_world)[:3, 3]) @ np.linalg.inv(np.array(rig.matrix_world)[:3, :3]).T
me.vertices.foreach_set('co', co.ravel()); me.update()
hair.name = me.name = 'tripo_hair'
for c in hair.users_collection: c.objects.unlink(hair)
human.users_collection[0].objects.link(hair)
head = next(b.name for b in rig.data.bones if b.name.lower().endswith('head') and 'top' not in b.name.lower())
vg = hair.vertex_groups.new(name=head); vg.add(range(n), 1.0, 'REPLACE')
hair.parent = rig
mod = hair.modifiers.new('Armature', 'ARMATURE'); mod.object = rig
m = me.materials[0]; m.name = 'tripo_hair'
bsdf = next(nd for nd in m.node_tree.nodes if nd.type == 'BSDF_PRINCIPLED')
bsdf.inputs['Roughness'].default_value = 0.55

hme = human.data
hmw = human.matrix_world
nv = len(hme.vertices)
sk = np.array([hmw @ v.co for v in hme.vertices])
eye_z = float(np.mean([(bpy.data.objects['Human.high-poly'].matrix_world @ v.co).z for v in bpy.data.objects['Human.high-poly'].data.vertices]))
top = sk[sk[:, 2] > eye_z - 0.1]
back_y = top[:, 1].min() + 0.76 * np.ptp(top[:, 1])            # the back of the ears
brow_z = eye_z + 0.045
ear_x = 0.8 * np.abs(top[:, 0]).max()
# in front of the ears the hair goes over the skin: a hair vertex on or under it comes out to LIFT above it
skin_bvh = BVHTree.FromPolygons([Vector(x) for x in sk], [p.vertices[:] for p in hme.polygons])
hw = np.array([hair.matrix_world @ Vector(c) for c in co])
lift = np.zeros_like(hw); held = np.zeros(n, bool)
for i, h in enumerate(hw):
    h = Vector(h)
    if h.y > back_y or h.z < eye_z - LIFT_BELOW:           # the head's front, down to the cheeks
        continue
    loc, nrm, _, d = skin_bvh.find_nearest(h, LIFT_REACH)
    if loc is not None and abs(loc.x) < ear_x and ((h - loc).dot(nrm) < 0 or d < LIFT):   # tufts stay off the ears
        lift[i] = np.array(loc + nrm * LIFT - h); held[i] = True
hedges = np.array([e.vertices[:] for e in me.edges])
for _ in range(3):                                         # a falloff ring, so the shell does not crease
    acc = np.zeros_like(lift); cnt = np.zeros(n)
    for a_, b_ in ((0, 1), (1, 0)):
        np.add.at(acc, hedges[:, a_], lift[hedges[:, b_]]); np.add.at(cnt, hedges[:, a_], 1)
    lift = np.where(held[:, None], lift, 0.5 * acc / np.maximum(cnt, 1)[:, None])
lifted = int(held.sum())
hw = hw + lift
rw = np.linalg.inv(np.array(hair.matrix_world))
co = hw @ rw[:3, :3].T + rw[:3, 3]
me.vertices.foreach_set('co', co.ravel()); me.update()
print('HAIR lifted off the skin', lifted)
bvh = BVHTree.FromPolygons([Vector(x) for x in hw], [p.vertices[:] for p in me.polygons])
off = np.zeros((nv, 3)); poke = np.zeros(nv, bool)
zmin = hw[:, 2].min()
for v in hme.vertices:
    p = hmw @ v.co
    if p.z < zmin or not (p.z > brow_z or p.y > back_y) or (abs(p.x) > ear_x and p.z < brow_z):   # only the scalp: never the face or ears
        continue
    nrm = (hmw.to_3x3() @ v.normal).normalized()
    clear = float(os.environ.get('SCALP_CLEAR', 0))        # a reshaped head's scalp can reach the hair between vertices: keep this far under it
    if clear and abs(p.x) > 0.045:
        near = bvh.find_nearest(p, clear)
        if near[0] is not None:
            off[v.index] = -(clear + 0.002 - near[3]) * np.array(nrm)
            poke[v.index] = True
            continue
    if bvh.ray_cast(p + nrm * 1e-4, nrm, 0.05)[0] is not None:
        continue                                   # hair over it already
    hit = bvh.ray_cast(p - nrm * 1e-4, -nrm, float(os.environ.get('POKE_REACH', 0.012)))
    if hit[0] is not None:
        off[v.index] = -(hit[3] + 0.004) * np.array(nrm)
        poke[v.index] = True
edges = np.array([e.vertices[:] for e in hme.edges])
for _ in range(4):                                 # spread a falloff ring, keep the pushed vertices' own offsets
    acc = np.zeros_like(off); cnt = np.zeros(nv)
    for a_, b_ in ((0, 1), (1, 0)):
        np.add.at(acc, edges[:, a_], off[edges[:, b_]]); np.add.at(cnt, edges[:, a_], 1)
    off = np.where(poke[:, None], off, 0.5 * acc / np.maximum(cnt, 1)[:, None])

loc = off @ np.linalg.inv(np.array(hmw)[:3, :3]).T          # world offset -> Human local
for kb in (hme.shape_keys.key_blocks if hme.shape_keys else []):
    k = np.zeros(nv * 3); kb.data.foreach_get('co', k); kb.data.foreach_set('co', (k.reshape(nv, 3) + loc).ravel())
v0 = np.zeros(nv * 3); hme.vertices.foreach_get('co', v0); hme.vertices.foreach_set('co', (v0.reshape(nv, 3) + loc).ravel())
hme.update()
print('HAIR scalp pushed', int(poke.sum()), 'max mm', round(float(np.linalg.norm(off, axis=1).max()) * 1000, 1))
print('HAIR', n, 'verts on bone', head, 'z range', co[:, 2].min().round(3), co[:, 2].max().round(3))
bpy.ops.wm.save_as_mainfile(filepath=out)
