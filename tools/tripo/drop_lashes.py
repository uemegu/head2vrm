"""Remove the 3D eyelashes object (the painted lashes of the Tripo skin are kept instead).   blender -b IN.blend -P drop_lashes.py -- OUT.blend"""
import sys
import bpy
out = sys.argv[sys.argv.index('--') + 1]
for o in [o for o in bpy.data.objects if o.type == 'MESH' and 'eyelash' in o.name.lower()]:
    print('DROP_LASHES', o.name)
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.wm.save_as_mainfile(filepath=out)
