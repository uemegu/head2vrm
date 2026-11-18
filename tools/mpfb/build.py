"""Build a realistic MPFB (MakeHuman) base in Blender and save it as base.blend.

blender --background --python build.py -- OUT_DIR
"""
import os
import sys

import bpy
from mathutils import Vector

from bl_ext.blender_org.mpfb.services import HumanService, TargetService, LocationService
from bl_ext.blender_org.mpfb.entities.clothes.mhclo import Mhclo
from bl_ext.blender_org.mpfb.services import ClothesService

OUT = sys.argv[sys.argv.index('--') + 1]
os.makedirs(OUT, exist_ok=True)

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)

info = HumanService.get_default_human_info() if hasattr(HumanService, 'get_default_human_info') else {}
info.update({
    'phenotype': {
        'gender': 0.0, 'age': 0.42, 'muscle': 0.45, 'weight': 0.4, 'proportions': 0.8,
        'height': 0.45, 'cupsize': 0.5, 'firmness': 0.5,
        'race': {'asian': 0.65, 'caucasian': 0.35, 'african': 0.0},
    },
    'rig': 'mixamo',
    'eyes': 'high-poly/high-poly.mhclo',
    'eyebrows': 'eyebrow001/eyebrow001.mhclo',
    'eyelashes': 'eyelashes01/eyelashes01.mhclo',
    'teeth': 'teeth_base/teeth_base.mhclo',
    'tongue': 'tongue01/tongue01.mhclo',
    'hair': 'ponytail01/ponytail01.mhclo',
    'clothes': ['female_casualsuit01/female_casualsuit01.mhclo', 'shoes01/shoes01.mhclo'],
    'skin_mhmat': 'young_asian_female/young_asian_female.mhmat',
    'skin_material_type': 'MAKESKIN',
    'eyes_material_type': 'MAKESKIN',
    'eyes_material_settings': {},
    'skin_material_settings': {},
    'alternative_materials': {},
    'targets': [],
    'expressions': [],
    'proxy': '',
})
settings = HumanService.get_default_deserialization_settings()
settings.update({'subdiv_levels': 0, 'override_skin_model': 'PRESET', 'override_rig': 'PRESET'})
basemesh = HumanService.deserialize_from_dict(info, settings)
rig = basemesh.parent
print('BASEMESH', basemesh.name, 'RIG', rig and rig.name)
for c in rig.children:
    print('CHILD', c.name, c.type, len(c.data.vertices) if c.type == 'MESH' else '')

# Bake the modelling (macro) shape keys into the mesh, so only expression keys get exported.
bpy.context.view_layer.objects.active = basemesh
basemesh.select_set(True)
bpy.ops.mpfb.bake_shapekeys()

# Expression units -> shape keys on the basemesh (MakeHuman's facial "units").
units = os.path.join(LocationService.get_mpfb_data('targets'), 'expression', 'units', 'asian')
KEYS = {
    'blink_l': [('eye-left-closure', 1.0)],
    'blink_r': [('eye-right-closure', 1.0)],
    'aa': [('mouth-open', 0.7)],
    'oh': [('mouth-open', 0.45), ('mouth-pursing', 0.6)],
    'ou': [('mouth-open', 0.15), ('mouth-protusion', 0.7), ('mouth-pursing', 0.5)],
    'ee': [('mouth-open', 0.2), ('mouth-retraction', 0.6)],
    'ih': [('mouth-open', 0.35), ('mouth-retraction', 0.3)],
    'smile': [('mouth-corner-puller', 0.8), ('eye-left-slit', 0.25), ('eye-right-slit', 0.25)],
    'angry': [('eyebrows-left-down', 0.8), ('eyebrows-right-down', 0.8), ('mouth-depression', 0.4)],
    'sad': [('eyebrows-left-inner-up', 0.8), ('eyebrows-right-inner-up', 0.8), ('mouth-depression', 0.5)],
    'surprised': [('eyebrows-left-up', 0.8), ('eyebrows-right-up', 0.8), ('eye-left-opened-up', 0.6),
                  ('eye-right-opened-up', 0.6), ('mouth-open', 0.35)],
}
if not basemesh.data.shape_keys:
    basemesh.shape_key_add(name='Basis', from_mix=False)
base_co = [v.co.copy() for v in basemesh.data.vertices]
for key, parts in KEYS.items():
    deltas = {}
    for unit, w in parts:
        tmp = TargetService.load_target(basemesh, os.path.join(units, unit + '.target.gz'), weight=0.0, name='tmp')
        for i, d in enumerate(tmp.data):
            off = d.co - base_co[i]
            if off.length_squared > 0:
                deltas[i] = deltas.get(i, Vector()) + off * w
        basemesh.shape_key_remove(tmp)
    kb = basemesh.shape_key_add(name=key, from_mix=False)
    for i, off in deltas.items():
        kb.data[i].co = base_co[i] + off
    print('KEY', key, len(deltas))

# Same keys on the fitted proxies (eyelashes, eyebrows, teeth...) through their MHCLO weights.
for child in rig.children:
    if child == basemesh or child.type != 'MESH':
        continue
    path = ClothesService.find_clothes_absolute_path(child)
    if not path:
        continue
    mhclo = Mhclo()
    mhclo.load(path)
    for kb_src in basemesh.data.shape_keys.key_blocks[1:]:
        offs = []
        for vi, m in mhclo.verts.items():
            if vi >= len(child.data.vertices):
                continue
            o = sum(((kb_src.data[v].co - base_co[v]) * w for v, w in zip(m['verts'], m['weights'])), Vector())
            if o.length > 1e-6:
                offs.append((vi, o))
        if not offs:
            continue
        if not child.data.shape_keys:
            child.shape_key_add(name='Basis', from_mix=False)
        kb = child.shape_key_add(name=kb_src.name, from_mix=False)
        for vi, o in offs:
            kb.data[vi].co = child.data.vertices[vi].co + o
    print('PROXY KEYS', child.name, child.data.shape_keys and len(child.data.shape_keys.key_blocks))

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'base.blend'))
print('SAVED')
