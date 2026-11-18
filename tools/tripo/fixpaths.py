"""Point MPFB images back at ~/Library (base_arkit.blend was saved one folder deeper), then save.

blender --background IN.blend --python fixpaths.py -- OUT_BLEND
"""
import os, sys
import bpy
out = sys.argv[sys.argv.index('--') + 1]
for im in bpy.data.images:
    if im.source == 'FILE' and '/Library/Application Support/' in im.filepath:
        im.filepath = os.path.expanduser('~/Library/Application Support/') + im.filepath.split('/Library/Application Support/')[1]
        im.reload()
        print('PATH', im.name, im.has_data)
bpy.ops.wm.save_as_mainfile(filepath=out)
