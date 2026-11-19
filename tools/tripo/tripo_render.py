"""Side-by-side renders: the input head (moved into the basemesh frame) and the warped MPFB head with the baked input skin.

blender --background warped.blend --python tripo_render.py -- TRIPO_IN_MPFB_NPZ TRIPO_BASECOLOR SKIN_PNG OUT_DIR
Writes OUT_DIR/r_<view>_<tripo|mpfb>.png. Same lights and cameras for both; hair cards, clothes and shoes hidden.
"""
import math, os, sys
import bpy, mathutils
import numpy as np

tri_npz, basecolor, skin, out = sys.argv[sys.argv.index('--') + 1:]
d = np.load(tri_npz)
me = bpy.data.meshes.new('tripo'); me.from_pydata(d['co'].tolist(), [], d['tris'].tolist())
me.uv_layers.new(name='UVMap').data.foreach_set('uv', d['tuv'].reshape(-1)); me.update()
for p in me.polygons: p.use_smooth = True
tri = bpy.data.objects.new('tripo', me); bpy.context.scene.collection.objects.link(tri)
mat = bpy.data.materials.new('tripo'); mat.use_nodes = True; nt = mat.node_tree
tx = nt.nodes.new('ShaderNodeTexImage'); tx.image = bpy.data.images.load(basecolor)
bsdf = nt.nodes['Principled BSDF']; bsdf.inputs['Roughness'].default_value = 0.5
nt.links.new(tx.outputs['Color'], bsdf.inputs['Base Color']); me.materials.append(mat)

# base_arkit.blend was made one folder deeper: its relative MPFB paths miss ~/Library
for im in bpy.data.images:
    if im.source == 'FILE' and '/Library/Application Support/' in im.filepath:
        im.filepath = os.path.expanduser('~/Library/Application Support/') + im.filepath.split('/Library/Application Support/')[1]
        im.reload()
# the saved shape-key values are all 1.0 (only the VRM export reads the keys): show the neutral face
for o in bpy.data.objects:
    if o.type == 'MESH' and o.data.shape_keys:
        for kb in o.data.shape_keys.key_blocks[1:]:
            kb.value = 0.0
body = bpy.data.objects['Human']
for s in body.material_slots:
    for n in s.material.node_tree.nodes:
        if n.type == 'TEX_IMAGE' and n.image and 'female_diffuse' in n.image.name.lower():
            n.image = bpy.data.images.load(skin)
if os.environ.get('EYE'):
    for s in bpy.data.objects['Human.high-poly'].material_slots:
        for n in s.material.node_tree.nodes:
            if n.type == 'TEX_IMAGE' and n.image and 'eye' in n.image.name.lower():
                n.image = bpy.data.images.load(os.environ['EYE'])
for o in bpy.data.objects:
    if o.type == 'MESH' and any(k in o.name for k in ('ponytail', 'suit', 'shoes')):
        o.hide_render = True

lm_head = np.array([0.0, -0.06, 1.30])  # rough head centre in the basemesh frame
cen = np.array([v.co for v in body.data.vertices])  # local == world for MPFB (identity transform assumed)
sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'; sc.view_settings.view_transform = 'AgX'
sc.render.resolution_x = sc.render.resolution_y = 1024
sc.render.film_transparent = False
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.25, 0.25, 0.27, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.5
for name, loc, energy in (('key', (-0.6, -1.0, 1.7), 40), ('fill', (0.8, -0.8, 1.4), 12), ('rim', (0.2, 0.9, 1.6), 25)):
    L = bpy.data.objects.new(name, bpy.data.lights.new(name, 'AREA')); L.data.energy = energy; L.data.size = 0.6
    L.location = loc; L.rotation_euler = (mathutils.Vector(lm_head) - mathutils.Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    sc.collection.objects.link(L)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 85
tco = d['co']; top = tco[:, 2].max()
head_c = np.array([0.0, tco[tco[:, 2] > top - 0.25][:, 1].mean(), top - 0.12])
views = [('front', 0, 0.0, 0.7, head_c), ('q35', -35, 0.0, 0.7, head_c), ('side', -90, 0.0, 0.7, head_c), ('back', 180, 0.0, 0.7, head_c),
         ('closeF', 0, 0.0, 0.32, head_c + np.array([0, 0, 0.0])), ('closeS', -60, 0.0, 0.32, head_c + np.array([0, 0, 0.0]))]
for name, ang, el, zoom, tgt in views:
    a = math.radians(ang)
    dist = 1.25 * zoom
    loc = mathutils.Vector((tgt[0] + math.sin(a) * dist, tgt[1] - math.cos(a) * dist, tgt[2]))
    cam.location = loc
    cam.rotation_euler = (mathutils.Vector(tgt) - loc).to_track_quat('-Z', 'Y').to_euler()
    for which in ('tripo', 'mpfb'):
        tri.hide_render = which != 'tripo'
        for o in bpy.data.objects:
            if o.type == 'MESH' and o.name.startswith('Human') and not any(k in o.name for k in ('ponytail', 'suit', 'shoes')):
                o.hide_render = which != 'mpfb'
        sc.render.filepath = os.path.join(out, f'r_{name}_{which}.png'); bpy.ops.render.render(write_still=True)
print('RENDERED')
