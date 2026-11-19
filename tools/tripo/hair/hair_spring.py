"""Make the grafted input hair (hair_graft.py) sway: bone chains under the head bone, the hair re-skinned to them,
and VRM 1.0 spring bones with head / neck colliders.

blender --background textured_hair.blend --python hair_spring.py -- OUT_BLEND [MODE]
MODE
  N (default 12)  loose short hair: N chains round the head at even azimuths. Each starts at the crown (ROOT_Z) and
                  follows the outer hair surface (INSET inside it) down to the hair's lower edge at that azimuth, in
                  3 swinging bones plus a short tip bone. A vertex is weighted along its azimuth's two nearest chains
                  by height (hat functions on the joints, the root joint = the head bone).
  updo[:CELLS]    hair tied up: the shell stays on the head; only the bun (the hair sticking out furthest from the
                  skull) gets one stiff bone, and the atlas cells CELLS (comma-separated, hairgraft/split.py cell
                  order: 0 = first hair part) are loose strands, each a soft 3-bone chain ordered by height.
"""
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
out = args[0]
MODE = args[1] if len(args) > 1 else '12'
INSET = 0.010          # ring joints this far inside the outer hair surface
SEG = 3                # swinging bones per chain
HIT = 0.008            # joint hit radius

rig = bpy.data.objects['Human.rig']
hair = bpy.data.objects['tripo_hair']
HEAD = next(g.name for g in hair.vertex_groups)
co = np.array([hair.matrix_world @ v.co for v in hair.data.vertices])
nv = len(co)
yc = (co[:, 1].min() + co[:, 1].max()) / 2


def hats(s, n):
    """(len(s), n+1) hat weights of a parameter s in [0, n] on the joints 0..n."""
    return np.stack([np.clip(1 - np.abs(s - j), 0, 1) for j in range(n + 1)], 1)


def adiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


# each chain: name, joints (SEG+1 points + tip), per-vertex weights (nv, SEG+1: column 0 = the head bone), spring
chains = []


def ring(nch):
    root_z = co[:, 2].max() - 0.025
    az = np.arctan2(co[:, 0], -(co[:, 1] - yc))                # 0 = front (-Y), +pi/2 = +X
    rad = np.hypot(co[:, 0], co[:, 1] - yc)
    step = 2 * math.pi / nch
    t = (az + math.pi) / step
    k0 = np.floor(t).astype(int) % nch
    f = t - np.floor(t)
    for k in range(nch):
        a = -math.pi + k * step
        m = np.abs(adiff(az, a)) < step * 0.6
        z_tip = float(np.percentile(co[m, 2], 3))
        d = np.array([math.sin(a), -math.cos(a), 0.0])
        joints = []
        for j in range(SEG + 1):
            z = root_z + (z_tip - root_z) * j / SEG
            band = m & (np.abs(co[:, 2] - z) < 0.008)
            r = float(rad[band].max()) if band.any() else float(rad[m].max())
            joints.append(np.array([0.0, yc, z]) + d * max(r - INSET, 0.02))
        blend = np.where(k0 == k, 1 - f, 0) + np.where((k0 + 1) % nch == k, f, 0)
        s = np.clip((root_z - co[:, 2]) / (root_z - z_tip), 0, 1) * SEG
        chains.append({'name': f'hair_{k:02d}', 'joints': joints, 'w': hats(s, SEG) * blend[:, None],
                       'stiff': [2.2 - 0.4 * j for j in range(SEG)], 'drag': 0.45, 'grav': 0.08})


def cell_mask(cells):
    """vertices whose UVs lie in split.py's atlas cells (first part 3/4 bottom-left, next 3 right column, rest 1/16 top strip)."""
    rects = [(0, 0, 0.75)] + [(0.75, 0.25 * k, 0.25) for k in range(3)] + \
            [(0.0625 * (k % 16), 0.75 + 0.0625 * (k // 16), 0.0625) for k in range(64)]
    me = hair.data
    uv = np.zeros(len(me.loops) * 2); me.uv_layers.active.data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
    lv = np.zeros(len(me.loops), int); me.loops.foreach_get('vertex_index', lv)
    m = np.zeros(nv, bool)
    for c in cells:
        x0, y0, w = rects[c]
        inside = (uv[:, 0] >= x0) & (uv[:, 0] < x0 + w) & (uv[:, 1] >= y0) & (uv[:, 1] < y0 + w)
        m[lv[inside]] = True
    return m


def updo(cells):
    strand = cell_mask(cells) if cells else np.zeros(nv, bool)
    shell = ~strand
    # bun: the direction (back half) in which the hair reaches furthest from the head centre
    c = np.array([0.0, yc, float(np.median(co[shell, 2]))])
    v = co - c
    r = np.linalg.norm(v, axis=1)
    back = shell & (v[:, 1] > 0)
    far = np.argsort(r[back])[-30:]
    d = v[back][far].mean(0); d /= np.linalg.norm(d)
    cosang = (v @ d) / np.maximum(r, 1e-6)
    proj = v @ d
    cone = shell & (cosang > math.cos(math.radians(40)))
    r1 = float(proj[cone].max())
    side = shell & (cosang > math.cos(math.radians(55))) & (cosang < math.cos(math.radians(45)))
    r0 = float(np.percentile(proj[side], 80) / math.cos(math.radians(50)))   # skull surface along d (from the cone's rim)
    r0 = min(r0, r1 - 0.02)
    s = np.clip((proj - r0) / (r1 - r0), 0, 1) * (cosang > math.cos(math.radians(60))) * shell
    w = np.zeros((nv, 2)); w[:, 1] = s * s * (3 - 2 * s); w[:, 0] = 1 - w[:, 1]
    j0, j1 = c + d * r0, c + d * (r1 - 0.005)
    chains.append({'name': 'hair_bun', 'joints': [j0, j1, j1 + d * 0.01], 'w': w, 'stiff': [3.5], 'drag': 0.6,
                   'grav': 0.12, 'seg': 1})
    print('SPRING bun dir', d.round(2), 'from', round(r0, 3), 'to', round(r1, 3), 'verts', int((s > 0.05).sum()))
    # strands: one chain per cell, ordered by height, top 15% held by the head
    for ci in cells:
        m = cell_mask([ci])
        z = co[m, 2]
        top, bot = float(z.max()), float(z.min())
        sp = np.clip((top - co[:, 2]) / (top - bot), 0, 1)
        sp = np.clip((sp - 0.15) / 0.85, 0, 1) * SEG
        joints = []
        for j in range(SEG + 1):
            zj = top - (top - bot) * (0.15 + 0.85 * j / SEG)
            near = m & (np.abs(co[:, 2] - zj) < (top - bot) / (2 * SEG) + 1e-3)
            joints.append(co[near].mean(0))
        w = hats(sp, SEG) * m[:, None]
        w[:, 0] += ~m
        chains.append({'name': f'hair_strand{ci}', 'joints': joints, 'w': w,
                       'stiff': [0.9, 0.7, 0.5], 'drag': 0.35, 'grav': 0.25})
        print('SPRING strand cell', ci, 'verts', int(m.sum()), 'length', round(top - bot, 3))
    # everything else: the shell on the head (column 0 of the bun chain already covers it)


if MODE.startswith('updo'):
    updo([int(x) for x in MODE.split(':')[1].split(',')] if ':' in MODE else [])
else:
    ring(int(MODE))

# bones
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='EDIT')
eb = rig.data.edit_bones
for c in chains:
    J = list(c['joints'])
    n = c.get('seg', SEG)
    if len(J) == n + 1:
        J.append(J[-1] + (J[-1] - J[-2]) * 0.3)
    parent = eb[HEAD]
    c['bones'] = []
    for j in range(n + 1):
        b = eb.new(f"{c['name']}_{j}")
        b.head, b.tail = Vector(J[j]), Vector(J[j + 1])
        b.parent = parent
        b.use_connect = j > 0
        b.use_deform = j < n
        parent = b
        c['bones'].append(b.name)
bpy.ops.object.mode_set(mode='OBJECT')

# weights: chains share the head column; normalise so each vertex sums to 1
cols = [c['w'] for c in chains]
head_w = sum(w[:, 0] for w in cols)
if MODE.startswith('updo'):     # every chain's head column already says "the rest stays on the head"
    head_w = np.min(np.stack([w[:, 0] for w in cols]), 0)
total = head_w + sum(w[:, 1:].sum(1) for w in cols)
for vg in list(hair.vertex_groups):
    hair.vertex_groups.remove(vg)
groups = [(HEAD, head_w)] + [(c['bones'][j], c['w'][:, j + 1]) for c in chains for j in range(c.get('seg', SEG))]
for name, w in groups:
    g = hair.vertex_groups.new(name=name)
    w = w / np.maximum(total, 1e-6)
    for i in np.nonzero(w > 1e-4)[0]:
        g.add([int(i)], float(w[i]), 'REPLACE')

# spring bones
ext = rig.data.vrm_addon_extension
ext.spec_version = '1.0'
sb = ext.spring_bone1
centre = Vector((0.0, yc, float(np.median(co[:, 2])) - 0.01))
inner = [J for c in chains for J in c['joints'][1:]] or [centre + Vector((0, 0, 0.08))]
r_head = min((Vector(J) - centre).length for J in inner) - HIT - 0.004
r_head = max(r_head, 0.04)


def add_sphere(bone, world, radius):
    bpy.ops.vrm.add_spring_bone1_collider(armature_object_name=rig.name)
    col = sb.colliders[-1]
    col.node.bone_name = bone
    col.shape_type = 'Sphere'
    col.shape.sphere.offset = rig.pose.bones[bone].matrix.inverted() @ world
    col.shape.sphere.radius = radius
    return col


neck = rig.data.bones['mixamorig:Neck']
cols = [add_sphere(HEAD, centre, r_head),
        add_sphere('mixamorig:Neck', (neck.head_local + neck.tail_local) / 2 + Vector((0, 0.01, 0)), 0.05)]
if 'mixamorig:Spine2' in rig.data.bones:     # shoulders / upper chest, for strands that reach the collar
    sp2 = rig.data.bones['mixamorig:Spine2']
    for x in (-0.09, 0.09):
        cols.append(add_sphere('mixamorig:Spine2', sp2.tail_local + Vector((x, 0.0, -0.02)), 0.06))
bpy.ops.vrm.add_spring_bone1_collider_group(armature_object_name=rig.name)
grp = sb.collider_groups[-1]
grp.vrm_name = 'head_neck'
for c in cols:
    bpy.ops.vrm.add_spring_bone1_collider_group_collider(armature_object_name=rig.name,
                                                        collider_group_index=len(sb.collider_groups) - 1)
    grp.colliders[-1].collider_uuid = c.uuid
for c in chains:
    bpy.ops.vrm.add_spring_bone1_spring(armature_object_name=rig.name)
    sp = sb.springs[-1]
    sp.vrm_name = c['name']
    for j, n in enumerate(c['bones']):
        jt = sp.joints.add()
        jt.node.bone_name = n
        jt.stiffness = c['stiff'][min(j, len(c['stiff']) - 1)]
        jt.drag_force = c['drag']
        jt.gravity_power = c['grav']
        jt.hit_radius = HIT
    sp.collider_groups.add().collider_group_uuid = grp.uuid
print('SPRING mode', MODE, 'chains', len(chains), 'bones', sum(len(c['bones']) for c in chains),
      'head collider r', round(r_head, 3), 'centre', [round(x, 3) for x in centre])
bpy.ops.wm.save_as_mainfile(filepath=out)
