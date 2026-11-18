"""Keep only the head: the body cut off at the neck and the cut closed, clothes, shoes and hair cards removed.

blender --background face_only.blend --python head_only.py -- OUT_BLEND

The skeleton stays whole (VRM needs its humanoid bones), so the head sits where it sat on the body and turns with the neck and
head bones; the expressions (the 52 ARKit keys and the presets made from them) stay on the head's meshes.
NECK_CUT (0.45): the cut's height between the Neck bone's root (0) and the Head bone's root (1).
NECK_TILT (12): degrees the cut leans, lower at the front under the chin than at the nape.
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

out = sys.argv[sys.argv.index('--') + 1]
KEEP = ('Human', 'high-poly', 'teeth', 'tongue', 'eyelash', 'eyebrow', 'caruncle', 'tripo_hair')
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    part = o.name.split('.', 1)[1] if '.' in o.name else o.name
    if not (o.name in ('Human', 'tripo_hair') or any(k in part for k in KEEP[1:])):
        print('HEAD_ONLY drop', o.name)
        bpy.data.objects.remove(o)

rig = bpy.data.objects['Human.rig']
bone = lambda *names: next(rig.data.bones[n] for n in names if n in rig.data.bones)
neck = rig.matrix_world @ bone('mixamorig:Neck', 'neck01', 'neck').head_local
head = rig.matrix_world @ bone('mixamorig:Head', 'head').head_local
c = neck.lerp(head, float(os.environ.get('NECK_CUT', 0.45)))
t = math.radians(float(os.environ.get('NECK_TILT', 12)))
n = Vector((0, -math.sin(t), math.cos(t)))   # up, leaning back: front (-Y) lower. Points on the head's side are kept.

hm = bpy.data.objects['Human']
mw = hm.matrix_world
co_l, no_l = mw.inverted() @ c, (mw.inverted().to_3x3() @ n).normalized()
bm = bmesh.new()
bm.from_mesh(hm.data)
before = len(bm.verts)
geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=co_l, plane_no=no_l, clear_inner=True)
cut = {e for e in res['geom_cut'] if isinstance(e, bmesh.types.BMEdge)}
# the new open rim: boundary edges lying on the plane (the eye and mouth openings are far above it)
rim = [e for e in bm.edges if e.is_boundary and all(abs((v.co - co_l).dot(no_l)) < 1e-4 for v in e.verts)]
print('HEAD_ONLY verts', before, '->', len(bm.verts), 'rim edges', len(rim))
filled = bmesh.ops.holes_fill(bm, edges=rim, sides=0)['faces']
tri = bmesh.ops.triangulate(bm, faces=filled)['faces'] if filled else []
# the cap's colour: every corner at the UV of the rim's frontmost point, one flat skin tone (seen only from below)
uv = bm.loops.layers.uv.active
if tri and uv:
    front = min((v for f in tri for v in f.verts), key=lambda v: v.co.y)
    ref = next(l[uv].uv.copy() for l in front.link_loops if l.face not in tri)
    for f in tri:
        for l in f.loops:
            l[uv].uv = ref
print('HEAD_ONLY cap faces', len(tri))
bm.to_mesh(hm.data)
bm.free()
hm.data.update()
# the shape keys of the vertices the cut made: bisect interpolates them; check none of them moves
sk = hm.data.shape_keys
if sk:
    base = sk.key_blocks[0]
    worst = 0.0
    for kb in sk.key_blocks[1:]:
        for i, v in enumerate(hm.data.vertices):
            if abs((mw @ v.co - c).dot(n)) < 2e-3:
                worst = max(worst, (kb.data[i].co - base.data[i].co).length)
    print('HEAD_ONLY rim moves under any key at most %.2f mm' % (worst * 1000))
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
