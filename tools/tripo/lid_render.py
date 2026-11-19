"""Close orthographic front renders of both eyes, unlit (the textures' own colours), all from one camera:
  <out>_tripo.png  the input head (its mesh in the MPFB frame, tripo_in_mpfb.npz + tripo_basecolor.png)
  <out>_mpfb.png   the fitted MPFB head (Human, eyeballs, lashes)
  <out>_mask.png   the same with the eyeballs pure magenta and no lashes: the visible opening
blender -b X.blend -P eye_render.py -- OUT_PREFIX TRIPO_NPZ TRIPO_PNG   (X.blend: warped_e0/textured; its shape keys are set to 0)
Camera: centred between the eyeballs, 2 px per 0.05 mm (RES px over WIDTH m); VIEW=tripo: the viewer's input-model view instead.
MASK_ONLY=1 TRIPO_TOO=1: the input head and the mask from one camera; INNER=1: the lids' inner faces cyan in the mask. <out>_cam.json: the camera."""
import bpy, sys, mathutils, numpy as np
out, tnpz, tpng = sys.argv[sys.argv.index('--') + 1:]
import os
RES_X, RES_Y, WIDTH = (2000, 2000, 0.20) if os.environ.get("FACE") else (2000, 700, 0.10)
sc = bpy.context.scene
for o in sc.objects:
    if o.type == 'MESH' and o.data.shape_keys:
        for kb in o.data.shape_keys.key_blocks[1:]: kb.value = 0.0
    if o.type == 'MESH':
        o.hide_render = o.name not in ('Human', 'Human.high-poly', 'Human.eyelashes01')
def unlit(m, color=None):
    nt = m.node_tree
    outn = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    em = nt.nodes.new('ShaderNodeEmission')
    if color is not None:
        em.inputs[0].default_value = color
        nt.links.new(em.outputs[0], outn.inputs['Surface']); return
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf or not bsdf.inputs['Base Color'].links: return
    nt.links.new(bsdf.inputs['Base Color'].links[0].from_socket, em.inputs[0]); sh = em
    if bsdf.inputs['Alpha'].links:
        tr = nt.nodes.new('ShaderNodeBsdfTransparent'); mx = nt.nodes.new('ShaderNodeMixShader')
        nt.links.new(bsdf.inputs['Alpha'].links[0].from_socket, mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2]); sh = mx
    nt.links.new(sh.outputs[0], outn.inputs['Surface'])
for o in sc.objects:
    if o.type == 'MESH' and not o.hide_render:
        for m in o.data.materials:
            if m and m.use_nodes: unlit(m)
dg = bpy.context.evaluated_depsgraph_get()
eye = sc.objects['Human.high-poly']
E = np.array([eye.matrix_world @ v.co for v in eye.evaluated_get(dg).data.vertices])
c = np.array([float(v) for v in os.environ["CAM_CENTRE"].split(",")]) if os.environ.get("CAM_CENTRE") else E.mean(0)
# the input's own head
d = np.load(tnpz)
me = bpy.data.meshes.new('tri'); me.from_pydata(d['co'].tolist(), [], d['tris'].tolist()); me.update()
uvl = me.uv_layers.new()
for poly in me.polygons:
    for k, li in enumerate(poly.loop_indices): uvl.data[li].uv = d['tuv'][poly.index][k]
tri = bpy.data.objects.new('tri', me); sc.collection.objects.link(tri)
mat = bpy.data.materials.new('t'); mat.use_nodes = True; nt = mat.node_tree
tx = nt.nodes.new('ShaderNodeTexImage'); tx.image = bpy.data.images.load(tpng); tx.interpolation = 'Cubic'
em = nt.nodes.new('ShaderNodeEmission'); nt.links.new(tx.outputs[0], em.inputs[0])
nt.links.new(em.outputs[0], nt.nodes['Material Output'].inputs['Surface']); me.materials.append(mat)
sc.render.engine = 'BLENDER_EEVEE'; sc.view_settings.view_transform = 'Standard'
sc.render.resolution_x, sc.render.resolution_y = RES_X, RES_Y
w = bpy.data.worlds.new('w'); sc.world = w; w.color = (0.1, 0.1, 0.1)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam
VIEW = os.environ.get('VIEW')   # 'tripo': the viewer's input-model view camera (viewer.js VIEWS.tripo): 7.94 x the pupils' spacing from
                                # the irises' front, aimed 0.59 x that spacing below the eyes, raised 10.4 deg, 23.6 deg high
if VIEW == 'tripo':
    k_, below, elev, fov = 7.94, 0.59, np.radians(float(os.environ.get('ELEV', 10.4))), np.radians(23.6)
    Lc, Rc = E[E[:, 0] > 0].mean(0), E[E[:, 0] < 0].mean(0)
    ipd = np.linalg.norm(Lc - Rc); mid = (Lc + Rc) / 2
    T = np.array([mid[0], E[:, 1].min(), mid[2] - below * ipd]); dist = k_ * ipd
    P = T + dist * np.array([0, -np.cos(elev), np.sin(elev)])
    cam.data.type = 'PERSP'; cam.data.sensor_fit = 'VERTICAL'; cam.data.angle = fov
    sc.render.resolution_x = sc.render.resolution_y = RES_X = RES_Y = int(os.environ.get('RES', 1600))
    cam.location = mathutils.Vector(P.tolist())
    cam.rotation_euler = (mathutils.Vector(T.tolist()) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    c = T
else:
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = WIDTH
    cam.location = mathutils.Vector((c[0], c[1] - 1.0, c[2])); cam.rotation_euler = (np.pi / 2, 0, 0)
bpy.context.view_layer.update()
def cam_json():
    import json
    M = np.array(cam.matrix_world)
    json.dump({'centre': c.tolist(), 'width_m': WIDTH, 'res': [RES_X, RES_Y], 'view': VIEW or 'ortho', 'cam_world': M.tolist(),
               'fov': cam.data.angle if VIEW else None, 'eye_front_y': float(E[:, 1].min())}, open(f'{out}_cam.json', 'w'))
def shot(name):
    sc.render.filepath = f'{out}_{name}.png'; bpy.ops.render.render(write_still=True)
human = [o for o in sc.objects if o.type == 'MESH' and o.name in ('Human', 'Human.high-poly', 'Human.eyelashes01')]
if os.environ.get('MASK_ONLY'):      # just the opening: skin black, eyeballs magenta, no lashes
    if os.environ.get('TRIPO_TOO'):  # and first the input head alone, from the same camera
        for o in human: o.hide_render = True
        tri.hide_render = False; shot('tripo')
        for o in human: o.hide_render = False
    tri.hide_render = True
    sc.objects['Human.eyelashes01'].hide_render = True
    for m in sc.objects['Human'].data.materials: unlit(m, (0, 0, 0, 1))
    for m in eye.data.materials: unlit(m, (1, 0, 1, 1))
    if os.environ.get('INNER'):      # the lids' inner faces (facing the eyeball) cyan: they are not skin either
        hum = sc.objects['Human']
        mi = bpy.data.materials.new('inner'); mi.use_nodes = True; unlit(mi, (0, 1, 1, 1))
        hum.data.materials.append(mi); k_in = len(hum.data.materials) - 1
        Mh = np.array(hum.matrix_world); cs = {s_: (lambda b: (b.min(0) + b.max(0)) / 2)(E[np.sign(E[:, 0]) == s_]) for s_ in (1, -1)}
        rad_ = (E[E[:, 0] > 0].max(0) - E[E[:, 0] > 0].min(0)).max() / 2
        for p in hum.data.polygons:
            pc = Mh[:3, :3] @ np.array(p.center) + Mh[:3, 3]; pn = Mh[:3, :3] @ np.array(p.normal)
            dd = pc - cs[1 if pc[0] > 0 else -1]; dist = np.linalg.norm(dd)
            if dist < rad_ + 0.004 and -(pn @ dd) / dist > 0.35:
                p.material_index = k_in
    shot('mask')
    cam_json()
    print('EYE_RENDER mask', c)
    raise SystemExit
for o in human: o.hide_render = True
tri.hide_render = False; shot('tripo')
tri.hide_render = True
for o in human: o.hide_render = False
shot('mpfb')
sc.objects['Human.eyelashes01'].hide_render = True
for m in eye.data.materials: unlit(m, (1, 0, 1, 1))
shot('mask')
import json
json.dump({'centre': c.tolist(), 'width_m': WIDTH, 'res': [RES_X, RES_Y]}, open(f'{out}_cam.json', 'w'))
print('EYE_RENDER', c, 'px per mm', RES_X / WIDTH / 1000)
