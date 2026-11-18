"""Add the 52 ARKit face units (MPFB's faceunits01 pack) as shape keys to base.blend, then save.

blender --background base.blend --python add_arkit.py -- OUT_BLEND
Keys go on the basemesh and, through the MHCLO weights, on the fitted proxies (eyes, lashes, teeth, tongue),
the same way build.py makes its expression keys. apply.py then carries them through the face warp.
"""
import sys

import bpy
from mathutils import Vector

from bl_ext.blender_org.mpfb.services import TargetService, ClothesService
from bl_ext.blender_org.mpfb.services.faceservice import ARKIT_FACEUNITS
from bl_ext.blender_org.mpfb.entities.clothes.mhclo import Mhclo

out = sys.argv[sys.argv.index('--') + 1]
basemesh = bpy.data.objects['Human']
rig = basemesh.parent
keys = basemesh.data.shape_keys.key_blocks
base_co = [d.co.copy() for d in keys[0].data]

added = []
for name in ARKIT_FACEUNITS:
    if name in keys:
        continue
    path = TargetService.target_full_path(name)
    if not path:
        raise SystemExit(f'target not found: {name} (is the faceunits01 pack installed?)')
    tmp = TargetService.load_target(basemesh, path, weight=0.0, name='tmp')
    deltas = {i: d.co - base_co[i] for i, d in enumerate(tmp.data) if (d.co - base_co[i]).length_squared > 0}
    basemesh.shape_key_remove(tmp)
    kb = basemesh.shape_key_add(name=name, from_mix=False)
    for i, off in deltas.items():
        kb.data[i].co = base_co[i] + off
    added.append(name)
print('ARKIT KEYS', len(added))

for child in rig.children:
    if child == basemesh or child.type != 'MESH':
        continue
    path = ClothesService.find_clothes_absolute_path(child)
    if not path:
        continue
    mhclo = Mhclo()
    mhclo.load(path)
    n = 0
    for name in added:
        src = keys[name]
        offs = []
        for vi, m in mhclo.verts.items():
            if vi >= len(child.data.vertices):
                continue
            o = sum(((src.data[v].co - base_co[v]) * w for v, w in zip(m['verts'], m['weights'])), Vector())
            if o.length > 1e-6:
                offs.append((vi, o))
        if not offs:
            continue
        if not child.data.shape_keys:
            child.shape_key_add(name='Basis', from_mix=False)
        kb = child.shape_key_add(name=name, from_mix=False)
        for vi, o in offs:
            kb.data[vi].co = child.data.vertices[vi].co + o
        n += 1
    print('PROXY ARKIT KEYS', child.name, n)

bpy.ops.wm.save_as_mainfile(filepath=out)
print('SAVED', out)
