"""New eyelashes grown along the visible lid margins, replacing MPFB's texture cards.   any face
blender -b IN.blend -P lash_new.py -- OUT.blend          (after the lid fit, instead of lash_seat.py; lash_blink.py can follow)
Why: MPFB's flat cards did not fit the fitted margins (the lower ones read as black blobs, the upper ones as a line across the eye).
The margin (the front-most vertices of the lid rim, per x, smoothed) is found as in lash_seat.py. On it, N thin tapered ribbons
(5 points each, width W0 mm at the root to a point) are rooted a hair in front of the margin: upper ones lean forward and up and
sweep outward (longer toward the outer corner), lower ones are short and lean down. The ribbons' width lies across the view
direction so each reads as one hair from the front. They use the lash material with every UV on one opaque black texel.
Every shape key of the object is rebuilt: a strand moves rigidly with the body's same-named key at the nearest margin vertex
(lash_blink.py then redoes the blink keys).
Env: UP_N LO_N (strands per eye), UP_LEN LO_LEN (mm, middle value), W0 LO_W0 (mm), ROOT_FWD (m), LASH_RGB (sRGB), LASH_ALPHA
LO_ALPHA, UP_LIFT UP_CURL. Before 2026-11-10 the image was linear (0.045, 0.032, 0.03): the same as sRGB 60,50,48 near enough.
"""
import os
import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index('--') + 1]
UP_N, LO_N = int(os.environ.get('UP_N', 34)), int(os.environ.get('LO_N', 24))
UP_LEN, LO_LEN = float(os.environ.get('UP_LEN', 3.6)) / 1000, float(os.environ.get('LO_LEN', 1.8)) / 1000
W0 = float(os.environ.get('W0', 0.13)) / 1000
LO_W0 = float(os.environ.get('LO_W0', os.environ.get('W0', 0.13))) / 1000
# colour (sRGB 0..255) and opacity; the lower ones may be fainter (two texels of the solid image). Defaults = before 2026-11-10
RGB = [float(v) for v in os.environ.get('LASH_RGB', '60,50,48').split(',')]
UP_A, LO_A = float(os.environ.get('LASH_ALPHA', 0.85)), float(os.environ.get('LO_ALPHA', os.environ.get('LASH_ALPHA', 0.85)))
UP_LIFT, UP_CURL = float(os.environ.get('UP_LIFT', 0.45)), float(os.environ.get('UP_CURL', 0.32))   # the upper ones' rise and its curl
UP_FWD, UP_BACK = float(os.environ.get('UP_FWD', 0.6)), float(os.environ.get('UP_BACK', 0.0))   # out of the lid first, then curling back
LO_OUT = float(os.environ.get('LO_OUT', 0.3))   # how far the lower ones lean out towards the temple at the outer corner
UP_CLUMP, LO_CLUMP, SWEEP = float(os.environ.get('UP_CLUMP', 0)), float(os.environ.get('LO_CLUMP', 0)), float(os.environ.get('SWEEP', 0))
ROOT_FWD = float(os.environ.get('ROOT_FWD', 0.0002))
SEG = 4
hum, lash, eye = (bpy.data.objects[n] for n in ('Human', 'Human.eyelashes01', 'Human.high-poly'))
get = lambda d, n: (lambda a: (d.foreach_get('co', a), a.reshape(n, 3))[1])(np.zeros(n * 3))
Mh, Ml, Me = (np.array(o.matrix_world) for o in (hum, lash, eye))
nh, ne = len(hum.data.vertices), len(eye.data.vertices)
V = get(hum.data.vertices, nh) @ Mh[:3, :3].T + Mh[:3, 3]
N = np.zeros(nh * 3); hum.data.vertices.foreach_get('normal', N); N = N.reshape(nh, 3) @ Mh[:3, :3].T
E = get(eye.data.vertices, ne) @ Me[:3, :3].T + Me[:3, 3]
ed = np.zeros(len(hum.data.edges) * 2, int); hum.data.edges.foreach_get('vertices', ed); ed = ed.reshape(-1, 2)
rng = np.random.default_rng(7)
unit = lambda v: v / max(np.linalg.norm(v), 1e-12)
UPV, FWD = np.array([0, 0, 1.0]), np.array([0, -1.0, 0])   # world: up is +z, the face looks toward -y (as lash_seat.py)

# the body's same-named keys, as world-space deltas, for the strands to ride on
bkeys = hum.data.shape_keys.key_blocks if hum.data.shape_keys else []
bdelta = {}
for k in bkeys:
    if k.name == bkeys[0].name:
        continue
    bdelta[k.name] = (get(k.data, nh) - get(bkeys[0].data, nh)) @ Mh[:3, :3].T
keynames = [k.name for k in (lash.data.shape_keys.key_blocks if lash.data.shape_keys else [])][1:]

verts, faces, root_of, lower_v = [], [], [], []
for sgn in (1, -1):
    e = E[np.sign(E[:, 0]) == sgn]; c = (e.min(0) + e.max(0)) / 2; r = (e.max(0) - e.min(0)).max() / 2
    d = V - c; dist = np.linalg.norm(d, axis=1); near = dist < r + 0.004
    s_ = (N * d).sum(1) / np.maximum(dist, 1e-9)
    inn, outer = near & (s_ < 0), near & (s_ >= 0) & (V[:, 1] < c[1])
    rim = np.unique(np.r_[ed[outer[ed[:, 0]] & inn[ed[:, 1]], 0], ed[outer[ed[:, 1]] & inn[ed[:, 0]], 1]])
    a, b = V[rim[np.argmin(V[rim, 0])]], V[rim[np.argmax(V[rim, 0])]]
    line = np.full(len(rim), c[2])   # upper / lower by the eyeball centre's height (the corners' chord misfiled one eye's lower margin as upper)
    OUT = np.array([sgn, 0, 0.0])                       # toward the temple
    for name, sel, n_, L0 in (('up', rim[V[rim, 2] >= line - 1e-4], UP_N, UP_LEN), ('lo', rim[V[rim, 2] < line + 1e-4], LO_N, LO_LEN)):
        xs = np.linspace(a[0], b[0], 30)
        P = []
        for x in xs:
            k = sel[np.abs(V[sel, 0] - x) < 0.0015]
            if len(k):
                P.append(V[k[np.argmin(V[k, 2]) if name == 'up' else np.argmax(V[k, 2])]])
        P = np.array(P); P = P[np.argsort(P[:, 0])]
        for _ in (1, 2):
            P[1:-1, 1:] = (P[:-2, 1:] + 2 * P[1:-1, 1:] + P[2:, 1:]) / 4
        outer_x = b[0] if sgn > 0 else a[0]; inner_x = a[0] if sgn > 0 else b[0]
        strands = []
        for i in range(n_):
            t = (i + 0.5) / n_                          # 0 at the outer corner side .. 1 at the inner one? (see u below)
            x = (a[0] + (b[0] - a[0]) * (0.14 + 0.84 * t) if sgn > 0 else a[0] + (b[0] - a[0]) * (0.02 + 0.84 * t))   # not into the inner corner (the white shows there)
            x = np.clip(x + rng.normal(0, 0.0003), P[0, 0], P[-1, 0])
            u = (x - inner_x) / (outer_x - inner_x)     # 0 inner corner .. 1 outer corner
            root = np.array([x, np.interp(x, P[:, 0], P[:, 1]), np.interp(x, P[:, 0], P[:, 2])]) + FWD * ROOT_FWD
            if name == 'up':
                L = L0 * (0.7 + 0.6 * u) * rng.uniform(0.7, 1.3)
                sweep = 0.6 * max(0.0, u - 0.75) / 0.25 * SWEEP      # the outermost ones lie out towards the temple
                d0 = unit(FWD * UP_FWD + UPV * UP_LIFT + OUT * (0.15 + 0.5 * u + sweep) + rng.normal(0, 0.14, 3))
                curl = UPV * UP_CURL - FWD * UP_BACK + OUT * 0.10
            else:
                L = L0 * (0.7 + 0.5 * u) * rng.uniform(0.7, 1.3)
                d0 = unit(FWD * 0.4 - UPV * 0.8 + OUT * (0.05 + LO_OUT * u) + rng.normal(0, 0.14, 3))
                curl = -UPV * 0.05 + OUT * 0.05
            strands.append([root, L, d0, curl, np.zeros(3)])
        # clumps (evenly spaced single lashes looked wrong): Tripo's lashes stand in groups of 2-4 whose tips meet. Neighbours are grouped and
        # each one's tip is drawn towards the group's mean tip by CLUMP (the pull grows with the square of the length along it)
        k_ = UP_CLUMP if name == 'up' else LO_CLUMP
        if k_ > 0:
            i = 0
            while i < len(strands):
                g = list(range(i, min(i + int(rng.integers(2, 5 if name == 'up' else 4)), len(strands))))
                tips = np.array([st[0] + st[1] * (st[2] + st[3]) for st in (strands[j] for j in g)])
                for j, tp in zip(g, tips):
                    strands[j][4] = k_ * (tips.mean(0) - tp)
                i = g[-1] + 1
        for root, L, d0, curl, pull in strands:
            pts = [root + L * (d0 * s + curl * s * s) + pull * s * s for s in np.linspace(0, 1, SEG + 1)]
            base = len(verts)
            for j, p in enumerate(pts):
                s = j / SEG
                tang = unit(pts[min(j + 1, SEG)] - pts[max(j - 1, 0)])
                wv = unit(np.cross(tang, np.array([0, 1.0, 0])))
                if np.linalg.norm(np.cross(tang, np.array([0, 1.0, 0]))) < 0.3:
                    wv = np.array([1.0, 0, 0])
                hw = (W0 if name == 'up' else LO_W0) / 2 * (1 - s) ** 0.8
                if j == SEG:
                    verts.append(p)
                else:
                    verts.append(p - wv * hw); verts.append(p + wv * hw)
            # ribbon: pairs (2j, 2j+1) for j<SEG, the tip at the end
            for j in range(SEG):
                i0 = base + 2 * j
                if j < SEG - 1:
                    faces.append((i0, i0 + 1, i0 + 3, i0 + 2))
                else:
                    faces.append((i0, i0 + 1, i0 + 2))
            root_of.append(np.argmin(np.linalg.norm(V - root, axis=1)))   # body vertex the strand rides on
            root_of.extend([root_of[-1]] * (2 * SEG))
            lower_v.extend([name == 'lo'] * (2 * SEG + 1))
V_new = np.array(verts)
rv = np.array(root_of[: 0] + root_of)
# root_of got one entry for the strand then 2*SEG copies: that is 2*SEG+1 per strand = vertices per strand
assert len(rv) == len(V_new), (len(rv), len(V_new))

# flip the faces so they face the camera (-y) at the front
fn = []
for f in faces:
    p = V_new[list(f)]
    fn.append(np.cross(p[1] - p[0], p[2] - p[0])[1])
if np.median(fn) > 0:
    faces = [tuple(reversed(f)) for f in faces]

# rebuild the object: geometry, one UV on an opaque black texel of the lash texture, the head group, every key
Mi = np.linalg.inv(Ml)
local = lambda W: (W @ Mi[:3, :3].T + Mi[:3, 3])
if lash.data.shape_keys:
    lash.shape_key_clear()
mesh = lash.data
mesh.clear_geometry()
mesh.from_pydata(local(V_new).tolist(), [], faces)
for uvl in list(mesh.uv_layers):
    mesh.uv_layers.remove(uvl)
uvl = mesh.uv_layers.new(name='UVMap')
lv = np.zeros(len(mesh.loops), int); mesh.loops.foreach_get('vertex_index', lv)
low = np.array(lower_v)[lv]
lin = [((c / 255 + 0.055) / 1.055) ** 2.4 if c / 255 > 0.04045 else c / 255 / 12.92 for c in RGB]
if os.environ.get('LASH_TEX', '0') == '1':
    # LASH_TEX=1 (2026-11-10): ribbons wider than a screen pixel with a hair drawn down their middle (opaque core CORE of the
    # width, soft sides, fading to the tip): 0.05 mm ribbons fell under a pixel in the viewer and showed as dotted lines.
    # u across the ribbon (0 left, 1 right, the tip 0.5), v along it (0 root .. 1 tip); upper ones in the left half, lower right.
    across = np.zeros(len(V_new)); along = np.zeros(len(V_new))
    per = 2 * SEG + 1
    for st in range(len(V_new) // per):
        for j in range(SEG):
            across[st * per + 2 * j] = 0.0; across[st * per + 2 * j + 1] = 1.0
            along[st * per + 2 * j] = along[st * per + 2 * j + 1] = j / SEG
        across[st * per + 2 * SEG] = 0.5; along[st * per + 2 * SEG] = 1.0
    uu = across[lv] * 0.5 + np.where(low, 0.5, 0.0); vv = 0.02 + 0.96 * along[lv]
    uvl.data.foreach_set('uv', np.c_[uu, vv].ravel())
    mesh.update()
    N_ = 64; CORE = float(os.environ.get('CORE', 0.35))
    ug, vg = np.meshgrid((np.arange(N_) + 0.5) / N_, (np.arange(N_) + 0.5) / N_)
    loc = (ug % 0.5) / 0.5                                       # 0..1 across within each half
    a = np.exp(-((loc - 0.5) / (CORE / 2)) ** 2 * 2.0) * (1 - 0.75 * vg ** 2)
    a *= np.where(ug < 0.5, UP_A, LO_A)
    px_ = np.zeros((N_, N_, 4)); px_[..., :3] = lin; px_[..., 3] = a
    img = bpy.data.images.new('lash_solid', N_, N_, alpha=True)
    img.pixels = px_.ravel().tolist()
else:
    uvl.data.foreach_set('uv', np.c_[np.where(low, 0.75, 0.25), np.full(len(lv), 0.5)].ravel())   # left half upper, right half lower
    mesh.update()
    img = bpy.data.images.new('lash_solid', 4, 4, alpha=True)
    img.pixels = [v for y in range(4) for x in range(4) for v in (*lin, UP_A if x < 2 else LO_A)]
img.pack()
for m in lash.data.materials:
    for nd in m.node_tree.nodes:
        if nd.type == 'TEX_IMAGE':
            nd.image = img
vg = lash.vertex_groups.get('mixamorig:Head') or lash.vertex_groups.new(name='mixamorig:Head')
vg.add(list(range(len(V_new))), 1.0, 'REPLACE')
for p in mesh.polygons:
    p.use_smooth = False
lash.shape_key_add(name='Basis', from_mix=False)
for name in keynames:
    kb = lash.shape_key_add(name=name, from_mix=False)
    dl = bdelta.get(name)
    W = V_new + (dl[rv] if dl is not None else 0)
    kb.data.foreach_set('co', local(W).ravel())
mesh.update()
print('LASH_NEW strands %d verts %d keys %d (body keys matched %d)' % (len(root_of) // (2 * SEG + 1), len(V_new), len(keynames), sum(n in bdelta for n in keynames)))
bpy.ops.wm.save_as_mainfile(filepath=out)
