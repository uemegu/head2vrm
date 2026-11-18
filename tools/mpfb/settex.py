"""Swap in the baked skin, dark hair and grey eyes, then save.

blender --background warped.blend --python settex.py -- SKIN_PNG HAIR_PNG OUT_BLEND
"""
import os, sys
import bpy
skin, hair, out = sys.argv[sys.argv.index('--') + 1:]
from bl_ext.blender_org.mpfb.services import LocationService
eye = os.environ.get('EYE', 'blue')
if not eye.endswith('.png'):
    eye = os.path.join(LocationService.get_user_data('eyes'), 'materials', eye + '_eye.png')
def swap(obj_part, match, path):
    for o in bpy.data.objects:
        if o.type != 'MESH' or obj_part not in o.name:
            continue
        for s in o.material_slots:
            for n in s.material.node_tree.nodes:
                if n.type == 'TEX_IMAGE' and n.image and match(n.image.name.lower()):
                    n.image = bpy.data.images.load(os.path.abspath(path), check_existing=True)
                    print('SWAP', o.name, s.material.name, '->', os.path.basename(path))
swap('Human', lambda n: 'diffuse' in n and 'male_diffuse' in n, skin)
swap('ponytail', lambda n: 'diffuse' in n, hair)
swap('high-poly', lambda n: 'eye' in n, eye)
bpy.ops.wm.save_as_mainfile(filepath=out)
