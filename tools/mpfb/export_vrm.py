"""Turn the textured MPFB character into a VRM 1.0 (humanoid from the Mixamo rig, expressions from shape keys).

blender --background textured.blend --python export_vrm.py -- OUT_VRM
"""
import os
import sys

import bpy

out = sys.argv[sys.argv.index('--') + 1]
rig = bpy.data.objects['Human.rig']
if os.environ.get('NO_LASHES'):   # rely on the lash line baked into the skin instead of the 3D cards
    for o in [o for o in bpy.data.objects if o.type == 'MESH' and 'eyelash' in o.name]:
        bpy.data.objects.remove(o)
if os.environ.get('NO_HAIR'):   # the hair is drawn some other way (e.g. Gaussians in the viewer)
    for o in [o for o in bpy.data.objects if o.type == 'MESH' and 'ponytail' in o.name]:
        bpy.data.objects.remove(o)

# glTF-friendly materials: image straight into Base Color / Alpha.
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    for s in o.material_slots:
        m = s.material
        nt = m.node_tree
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        tex = next(n for n in nt.nodes if n.type == 'TEX_IMAGE' and 'diffuse' in n.image.name.lower()
                   or n.type == 'TEX_IMAGE' and not any(k in n.image.name.lower() for k in ('_ao', '_normal')))
        nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        name = o.name.lower()
        if 'eyelash' in name or 'ponytail' in name or 'caruncle' in name:
            nt.links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
        elif 'high-poly' in name:
            # alpha through "greater than" -> the glTF exporter writes alphaMode MASK (cutout, depth-sorted)
            gt = nt.nodes.new('ShaderNodeMath')
            gt.operation = 'GREATER_THAN'
            gt.inputs[1].default_value = 0.5
            nt.links.new(tex.outputs['Alpha'], gt.inputs[0])
            nt.links.new(gt.outputs[0], bsdf.inputs['Alpha'])
        else:
            for l in list(bsdf.inputs['Alpha'].links):
                nt.links.remove(l)
            bsdf.inputs['Alpha'].default_value = 1.0
        bsdf.inputs['Roughness'].default_value = 1.0 if o.get('cloth') else 0.6   # cloth: tripo/outfit/fit_outfit.py
        rough_path = os.environ.get('ROUGHNESS', os.path.join(os.path.dirname(out), 'roughness_4k.png'))
        if o.name == 'Human' and os.path.exists(rough_path):
            # glTF metallicRoughness: G = roughness (glossy lips), metalness 0
            rt = nt.nodes.new('ShaderNodeTexImage')
            rt.image = bpy.data.images.load(rough_path, check_existing=True)
            rt.image.colorspace_settings.name = 'Non-Color'
            sep = nt.nodes.new('ShaderNodeSeparateColor')
            nt.links.new(rt.outputs['Color'], sep.inputs['Color'])
            nt.links.new(sep.outputs['Green'], bsdf.inputs['Roughness'])
            bsdf.inputs['Metallic'].default_value = 0.0
            print('ROUGHNESS', rough_path)
        if hasattr(m, 'surface_render_method'):
            m.surface_render_method = 'BLENDED' if ('eyelash' in name or 'caruncle' in name) else 'DITHERED'
        m.blend_method = 'BLEND' if ('eyelash' in name or 'caruncle' in name) else ('CLIP' if ('ponytail' in name or 'high-poly' in name) else 'OPAQUE')

# The eyes: the eyeball's broad sheen (roughness 0.6) laid a grey film over the dark iris and hid the pupil (seen on a dark-eyed test head).
# The eyeball now has no specular; the cornea (the outer shell, its faces on the texture's transparent texels) gets its own
# clear, glossy material (glTF KHR_materials_transmission + ior), so the reflections are sharp and sit on a smooth surface.
# CORNEA=0 keeps the old single material.
import numpy as np
if os.environ.get('CORNEA', '1') != '0':
    for o in [o for o in bpy.data.objects if o.type == 'MESH' and 'high-poly' in o.name.lower()]:
        m = o.material_slots[0].material; nt = m.node_tree
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        img = next(n.image for n in nt.nodes if n.type == 'TEX_IMAGE' and n.image)
        W, H = img.size; px = np.empty(W * H * 4, np.float32); img.pixels.foreach_get(px); alpha = px.reshape(H, W, 4)[..., 3]
        bsdf.inputs['Roughness'].default_value = 1.0; bsdf.inputs['Specular IOR Level'].default_value = 0.0
        cm = bpy.data.materials.new('cornea'); cm.use_nodes = True
        cb = next(n for n in cm.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        cb.inputs['Base Color'].default_value = (1, 1, 1, 1); cb.inputs['Roughness'].default_value = float(os.environ.get('CORNEA_ROUGH', 0.04))
        cb.inputs['Transmission Weight'].default_value = 1.0; cb.inputs['IOR'].default_value = 1.376; cb.inputs['Metallic'].default_value = 0.0
        o.data.materials.append(cm); ci = len(o.data.materials) - 1
        uvl = o.data.uv_layers.active.data; n_c = 0
        for p in o.data.polygons:
            a = [alpha[min(H - 1, int(uvl[li].uv[1] % 1 * H)), min(W - 1, int(uvl[li].uv[0] % 1 * W))] for li in p.loop_indices]
            if np.mean(a) < 0.5:
                p.material_index = ci; p.use_smooth = True; n_c += 1
        print('EXPORT cornea faces %d of %d on their own glossy material' % (n_c, len(o.data.polygons)))

ext = rig.data.vrm_addon_extension
ext.spec_version = '1.0'
bpy.context.view_layer.objects.active = rig
if not os.environ.get('KEEP_HUMANOID'):   # KEEP_HUMANOID=1: the blend already names its humanoid bones (e.g. an MHR skeleton)
    bpy.ops.vrm.assign_vrm1_humanoid_human_bones_automatically(armature_object_name=rig.name)
hb = ext.vrm1.humanoid.human_bones
print('HIPS', hb.hips.node.bone_name, 'HEAD', hb.head.node.bone_name, 'LHAND', hb.left_hand.node.bone_name)

meta = ext.vrm1.meta
meta.vrm_name = os.environ.get('VRM_NAME', 'face')
if not meta.authors:
    meta.authors.add()
meta.authors[0].value = os.environ.get('VRM_AUTHOR', 'unknown')

meshes = [o for o in bpy.data.objects if o.type == 'MESH' and o.data.shape_keys]
preset = ext.vrm1.expressions.preset
MAP = {
    'blink': ['blink_l', 'blink_r'], 'blink_left': ['blink_l'], 'blink_right': ['blink_r'],
    'aa': ['aa'], 'ih': ['ih'], 'ou': ['ou'], 'ee': ['ee'], 'oh': ['oh'],
    'happy': ['smile'], 'angry': ['angry'], 'sad': ['sad'], 'surprised': ['surprised'],
}
# With the ARKit face units present, the presets are mixes of them instead (the older MakeHuman-unit keys
# pushed the inside of the mouth out through the cheeks on "ih" / "oh").
ARKIT_MAP = {
    'blink': {'eyeBlinkLeft': 1, 'eyeBlinkRight': 1}, 'blink_left': {'eyeBlinkLeft': 1}, 'blink_right': {'eyeBlinkRight': 1},
    'aa': {'jawOpen': 0.55, 'mouthLowerDownLeft': 0.2, 'mouthLowerDownRight': 0.2},
    'ih': {'jawOpen': 0.15, 'mouthStretchLeft': 0.45, 'mouthStretchRight': 0.45, 'mouthSmileLeft': 0.2, 'mouthSmileRight': 0.2},
    'ou': {'jawOpen': 0.1, 'mouthPucker': 0.9, 'mouthFunnel': 0.2},
    'ee': {'jawOpen': 0.3, 'mouthStretchLeft': 0.35, 'mouthStretchRight': 0.35, 'mouthUpperUpLeft': 0.2, 'mouthUpperUpRight': 0.2},
    'oh': {'jawOpen': 0.35, 'mouthFunnel': 0.7},
    'happy': {'mouthSmileLeft': 0.8, 'mouthSmileRight': 0.8, 'cheekSquintLeft': 0.5, 'cheekSquintRight': 0.5, 'eyeSquintLeft': 0.3, 'eyeSquintRight': 0.3},
    'angry': {'browDownLeft': 0.8, 'browDownRight': 0.8, 'mouthFrownLeft': 0.3, 'mouthFrownRight': 0.3, 'noseSneerLeft': 0.2, 'noseSneerRight': 0.2},
    'sad': {'browInnerUp': 0.8, 'mouthFrownLeft': 0.5, 'mouthFrownRight': 0.5},
    'surprised': {'browInnerUp': 0.6, 'browOuterUpLeft': 0.6, 'browOuterUpRight': 0.6, 'eyeWideLeft': 0.6, 'eyeWideRight': 0.6, 'jawOpen': 0.25},
}
body_keys = bpy.data.objects['Human'].data.shape_keys.key_blocks
use_arkit = all(k in body_keys for m in ARKIT_MAP.values() for k in m)
if use_arkit:   # drop the older keys: left unbound, they would export with a default weight and stay applied
    old = {k for ks in MAP.values() for k in ks}
    for o in meshes:
        for kb in list(o.data.shape_keys.key_blocks):
            if kb.name in old:
                o.shape_key_remove(kb)
    print('REMOVED old keys', sorted(old))
for exp_name, keys in (ARKIT_MAP if use_arkit else {n: {k: 1.0 for k in ks} for n, ks in MAP.items()}).items():
    exp = getattr(preset, exp_name)
    for o in meshes:
        for k, wgt in keys.items():
            if k in o.data.shape_keys.key_blocks:
                b = exp.morph_target_binds.add()
                b.node.mesh_object_name = o.name
                b.index = k
                b.weight = wgt
    print('EXP', exp_name, len(exp.morph_target_binds), 'arkit' if use_arkit else '')
preset.blink.is_binary = False

if os.environ.get('LOOKAT_EXPR') and use_arkit:   # the gaze through the eyeLook keys: VRM's lookAt works in any app
    LOOK = {'look_up': {'eyeLookUpLeft': 1, 'eyeLookUpRight': 1}, 'look_down': {'eyeLookDownLeft': 1, 'eyeLookDownRight': 1},
            'look_left': {'eyeLookOutLeft': 1, 'eyeLookInRight': 1}, 'look_right': {'eyeLookInLeft': 1, 'eyeLookOutRight': 1}}
    for exp_name, keys in LOOK.items():
        exp = getattr(preset, exp_name)
        exp.morph_target_binds.clear()
        for o in meshes:
            for k, wgt in keys.items():
                if k in o.data.shape_keys.key_blocks:
                    b = exp.morph_target_binds.add()
                    b.node.mesh_object_name = o.name
                    b.index = k
                    b.weight = wgt
    la = ext.vrm1.look_at
    la.type = 'expression'
    for rm in (la.range_map_horizontal_inner, la.range_map_horizontal_outer, la.range_map_vertical_down, la.range_map_vertical_up):
        rm.input_max_value, rm.output_scale = float(os.environ.get('LOOK_MAX_DEG', 25)), 1.0
    eyes = [o for o in bpy.data.objects if o.type == 'MESH' and 'high-poly' in o.name]
    if eyes:   # the gaze from between the eyes, as an offset from the head bone (VRM: the head bone's space, glTF axes)
        from mathutils import Vector
        vs = [eyes[0].matrix_world @ v.co for v in eyes[0].data.vertices]
        mid = sum(vs, Vector()) / len(vs)
        hb_head = rig.matrix_world @ rig.data.bones[hb.head.node.bone_name].head_local
        d = mid - hb_head
        la.offset_from_head_bone = (d.x, d.z, -d.y)   # glTF axes (Y up, Z forward): the add-on writes the value as it is
    print('LOOKAT expression', [len(getattr(preset, n).morph_target_binds) for n in LOOK])

# ARKit face units (perfect sync): one custom expression per key, named as ARKit names it
ARKIT = [k.name for k in bpy.data.objects['Human'].data.shape_keys.key_blocks
         if k.name[:1].islower() and any(k.name.startswith(p) for p in ('brow', 'cheek', 'eye', 'jaw', 'mouth', 'nose', 'tongue'))]
custom = ext.vrm1.expressions.custom
for name in ARKIT:
    exp = custom.add()
    exp.custom_name = name
    for o in meshes:
        if name in o.data.shape_keys.key_blocks:
            b = exp.morph_target_binds.add()
            b.node.mesh_object_name = o.name
            b.index = name
            b.weight = 1.0
print('ARKIT EXPRESSIONS', len(ARKIT))

res = bpy.ops.export_scene.vrm(filepath=out)
print('EXPORT', res)
