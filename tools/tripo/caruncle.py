"""A caruncle in each inner corner: a small surface just in front of the eyeball, in the inner end of the opening, carrying the pink
the input texture shows there (projected from the lid-fit camera).   any face, after run.sh with LID_FIT
blender -b IN.blend -P caruncle.py -- OUT_DIR OUT.blend

Why: the input texture can paint the inner 2.5-3 mm of each opening pink (caruncle and plica) and the white begins ~3.5 mm from
the tip. With the opening to the tip (so the white does not shrink) the base mesh's eyeball showed white there; cutting the
opening short at the white (CARUNCLE in lid_arap.py) made the eye look small. The pink belongs there, as in the input's corner.
How: in the lid-fit camera image of the input head (lidfit/warp/v0_tripo.png) the pixels inside the approved outline (lidfit/check_lines.txt)
within REACH mm of the inner tip are the caruncle's picture; their opacity is their pinkness (a* from A0, full A0+A1) fading out
past FADE of REACH. A grid over them (STEP px) is lifted along the camera rays onto the eyeball (a sphere fitted to the eye mesh's
back half) plus LIFT (1.4 mm: on the ball itself it hid behind the lids' margins), so from the fit camera it sits exactly where the input texture shows it. It rides the head bone like the eyeball.
"""
import json
import os
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index('--') + 1:]
O, out = args[0], args[1]
E = lambda k, d: float(os.environ.get(k, d))
REACH, FADE, A0, A1, LIFT, STEP = E('REACH', 3.4), E('FADE', 1.0), E('A0', 8), E('A1', 8), E('LIFT', 0.0014), int(E('STEP', 2))

cam = json.load(open(f'{O}/lidfit/warp/v0_cam.json'))
Mw = np.array(cam['cam_world']); Rc, Cc = Mw[:3, :3], Mw[:3, 3]
RES = cam['res'][1]; F = (RES / 2) / np.tan(cam['fov'] / 2)
src = bpy.data.images.load(f'{O}/lidfit/warp/v0_tripo.png')
W_, H_ = src.size
pix = np.array(src.pixels[:], np.float32).reshape(H_, W_, 4)[::-1]          # top row first, like the camera's image
rgb = pix[..., :3]


def srgb_to_lab_a(c):
    """a* of sRGB colours in 0..1 (enough for 'how pink')."""
    lin = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return 500 * (f[..., 0] - f[..., 1])


A = srgb_to_lab_a(rgb)
mm_px = np.linalg.norm(Cc - np.array(cam['centre'])) / F * 1000

eye = bpy.data.objects['Human.high-poly']
EV = np.array([eye.matrix_world @ v.co for v in eye.data.vertices])
ln = [l for l in open(f'{O}/lidfit/check_lines.txt').read().split('\n') if l.strip()]
rig = bpy.data.objects['Human.rig']
for k in range(0, len(ln), 3):
    x0, x1 = map(float, ln[k].split())
    up, lo = np.array(ln[k + 1].split(), float), np.array(ln[k + 2].split(), float)
    xx = np.linspace(x0, x1, len(up))
    tip_left = abs(x0 - RES / 2) < abs(x1 - RES / 2)
    tip = x0 if tip_left else x1
    # the eyeball on this side: the one whose centre projects nearer this opening
    best = None
    for sgn in (1, -1):
        b = EV[np.sign(EV[:, 0]) == sgn]; bb = b[b[:, 1] > np.median(b[:, 1])]
        s_ = np.linalg.lstsq(np.c_[2 * bb, np.ones(len(bb))], (bb ** 2).sum(1), rcond=None)[0]
        c = s_[:3]; r = float(np.sqrt(s_[3] + c @ c))
        q = (c - Cc) @ Rc; cx = RES / 2 + F * q[0] / -q[2]
        if best is None or abs(cx - (x0 + x1) / 2) < best[0]:
            best = (abs(cx - (x0 + x1) / 2), c, r)
    _, c, r = best
    # the grid over the caruncle's pixels
    xs = np.arange(int(tip - REACH / mm_px) - 2 if not tip_left else int(tip) - 2, (int(tip) + 3) if not tip_left else int(tip + REACH / mm_px) + 3, STEP)
    ys = np.arange(int(min(up.min(), lo.min())) - 4, int(max(up.max(), lo.max())) + 5, STEP)
    gx, gy = np.meshgrid(xs, ys)
    yu, yl = np.interp(gx, xx, up), np.interp(gx, xx, lo)
    dist_mm = np.abs(gx - tip) * mm_px
    inside = (gy >= yu - 1) & (gy <= yl + 1) & (dist_mm <= REACH)
    gxi, gyi = np.clip(gx, 0, W_ - 1).astype(int), np.clip(gy, 0, H_ - 1).astype(int)
    alpha = np.clip((A[gyi, gxi] - A0) / A1, 0, 1) * np.clip((REACH - dist_mm) / FADE, 0, 1) * inside
    # rays onto the ball (+ LIFT)
    d = (Rc @ np.stack([(gx - RES / 2) / F, -(gy - RES / 2) / F, -np.ones_like(gx, dtype=float)], 0).reshape(3, -1)).T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    oc = Cc - c; bq = (d * oc).sum(1); cq = (oc ** 2).sum() - (r + LIFT) ** 2; disc = bq ** 2 - cq
    t = -bq - np.sqrt(np.maximum(disc, 0))
    P = Cc + d * t[:, None]
    hit = (disc > 0).reshape(gx.shape)
    keep_v = hit & (inside | np.roll(inside, 1, 0) | np.roll(inside, -1, 0) | np.roll(inside, 1, 1) | np.roll(inside, -1, 1))
    nyy, nxx = gx.shape
    vid = -np.ones(gx.shape, int); vid[keep_v] = np.arange(keep_v.sum())
    verts = P.reshape(nyy, nxx, 3)[keep_v]
    faces = []
    for i in range(nyy - 1):
        for j in range(nxx - 1):
            q4 = (vid[i, j], vid[i, j + 1], vid[i + 1, j + 1], vid[i + 1, j])
            if min(q4) >= 0 and (alpha[i:i + 2, j:j + 2] > 0.01).any():
                faces.append(q4)
    # texture: the crop of the picture with the pinkness as alpha
    tx0, tx1, ty0, ty1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    crop = rgb[ty0:ty1, tx0:tx1]
    cxg, cyg = np.meshgrid(np.arange(tx0, tx1), np.arange(ty0, ty1))
    a_crop = np.clip((A[cyg, cxg] - A0) / A1, 0, 1) * np.clip((REACH - np.abs(cxg - tip) * mm_px) / FADE, 0, 1)
    a_crop *= (cyg >= np.interp(cxg, xx, up) - 1) & (cyg <= np.interp(cxg, xx, lo) + 1)
    # soften its edge (a* falls off sharply where the input's white begins; the patch's own edge read as a cut line)
    pad = np.pad(a_crop, 2, mode='edge'); a_soft = sum(pad[i:i + a_crop.shape[0], j:j + a_crop.shape[1]] for i in range(5) for j in range(5)) / 25
    a_crop = np.minimum(a_crop, a_soft) * 0.5 + a_soft * 0.5
    tex = np.dstack([crop, a_crop])[::-1]                                      # bottom row first for Blender
    img = bpy.data.images.new(f'caruncle_{k // 3}', tx1 - tx0, ty1 - ty0, alpha=True)
    img.pixels.foreach_set(tex.ravel().astype(np.float32))
    img.filepath_raw = f'{O}/caruncle_{k // 3}.png'; img.file_format = 'PNG'; img.save(); img.pack()
    me = bpy.data.meshes.new(f'caruncle_{k // 3}')
    me.from_pydata(verts.tolist(), [], faces)
    uvl = me.uv_layers.new(name='UVMap')
    uv_v = np.c_[(gx[keep_v] - tx0 + 0.5) / (tx1 - tx0), 1 - (gy[keep_v] - ty0 + 0.5) / (ty1 - ty0)]
    li = np.zeros(len(me.loops), int); me.loops.foreach_get('vertex_index', li)
    uvl.data.foreach_set('uv', uv_v[li].ravel())
    me.update()
    # face the camera
    if len(me.polygons) and np.mean([p.normal.dot(__import__('mathutils').Vector((Cc - np.array(p.center)).tolist())) for p in me.polygons]) < 0:
        for p in me.polygons: p.flip()
    mat = bpy.data.materials.new(f'caruncle_{k // 3}'); mat.use_nodes = True; nt = mat.node_tree
    tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = img
    bs = nt.nodes['Principled BSDF']; nt.links.new(tn.outputs['Color'], bs.inputs['Base Color']); nt.links.new(tn.outputs['Alpha'], bs.inputs['Alpha'])
    bs.inputs['Roughness'].default_value = 0.35
    mat.blend_method = 'BLEND'
    me.materials.append(mat)
    ob = bpy.data.objects.new(f'Human.caruncle{k // 3}', me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = rig
    vg = ob.vertex_groups.new(name='mixamorig:Head'); vg.add(list(range(len(verts))), 1.0, 'REPLACE')
    mod = ob.modifiers.new('Armature', 'ARMATURE'); mod.object = rig
    print('CARUNCLE eye %d: %d verts, %d faces, opaque px %d' % (k // 3, len(verts), len(faces), int((a_crop > 0.5).sum())))
bpy.ops.wm.save_as_mainfile(filepath=out)
