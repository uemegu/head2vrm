"""Point the base's textures at this machine's MPFB data (they are MPFB's own CC0 images, not shipped here), then save.
blender --background tools/mpfb/base_asian/base_arkit.blend --python tools/relink_base.py -- MPFB_DATA
(setup.sh copies the shipped base_arkit.dist.blend to base_arkit.blend first, so the shipped file never holds a machine's paths)
"""
import os, sys
import bpy
data = sys.argv[sys.argv.index('--') + 1]
bad = 0
for im in bpy.data.images:
    if im.source != 'FILE' or '/mpfb/data/' not in im.filepath.replace('\\', '/'):
        continue
    p = os.path.join(data, im.filepath.replace('\\', '/').split('/mpfb/data/')[1])
    im.filepath = p
    ok = os.path.exists(p); bad += not ok
    print('RELINK', 'ok ' if ok else 'MISSING', p)
bpy.ops.wm.save_mainfile()
print('RELINK done, missing', bad)
