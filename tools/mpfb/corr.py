"""Landmarks seen on the base front render -> points on the basemesh surface (vertex triplets + barycentrics).

blender --background base.blend --python corr.py -- CAMS_JSON MARKS_NPY OUT_NPZ
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform

cams_path, marks_path, out_path = sys.argv[sys.argv.index('--') + 1:]
cam = json.load(open(cams_path))[0]
marks = np.load(marks_path)

body = bpy.data.objects['Human']
mw = body.matrix_world
mesh = body.data
gi = body.vertex_groups['body'].index
in_body = np.zeros(len(mesh.vertices), bool)
for v in mesh.vertices:
    in_body[v.index] = any(g.group == gi and g.weight > 0.5 for g in v.groups)
co = np.array([mw @ v.co for v in mesh.vertices])
tris = []
for poly in mesh.polygons:
    vs = list(poly.vertices)
    if not in_body[vs].all():
        continue
    for k in range(1, len(vs) - 1):
        tris.append((vs[0], vs[k], vs[k + 1]))
tris = np.array(tris)
bvh = BVHTree.FromPolygons([Vector(p) for p in co], tris.tolist())

M = Matrix(cam['matrix_world'])
size = cam['size']
f = cam['lens'] / cam['sensor'] * size
origin = M.translation
ids, trip, bary = [], [], []
for i, (u, v, _) in enumerate(marks):
    d_local = Vector(((u - size / 2) / f, -(v - size / 2) / f, -1.0))
    d = (M.to_3x3() @ d_local).normalized()
    hit, nrm, fi, dist = bvh.ray_cast(origin, d)
    if hit is None:
        continue
    a, b, c = tris[fi]
    w = barycentric_transform(hit, Vector(co[a]), Vector(co[b]), Vector(co[c]),
                              Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))
    ids.append(i)
    trip.append((a, b, c))
    bary.append(tuple(w))
np.savez(out_path, ids=np.array(ids), trip=np.array(trip), bary=np.array(bary), co=co, in_body=in_body, tris=tris)
print('CORR', len(ids))
