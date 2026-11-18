"""Make the blink keys fold the upper lashes down, then save.

blender --background X.blend --python lash_blink.py -- OUT_BLEND
MPFB carries the lashes through a blink by translation only, so on a closed eye the upper lashes still
point up and come out through the lid. For each blink key, every lash vertex of the closing eye is re-placed as:
its root (nearest point on the lid margin, carried to the same spot of the closed lid) plus its
rest offset from the root (lower lashes); the upper lashes are sunk into the closed lid.
Lower lashes are shortened to LOWER_KEEP of their length (the closed upper lid would hide most of them).
"""
import os
import sys

import bpy
from mathutils import Vector

out = sys.argv[sys.argv.index('--') + 1]
LOWER_KEEP = 0.2
CLOSED_SINK = float(os.environ.get('CLOSED_SINK', 0.3))   # closed upper lashes: sunk this share of the way to the eye centre
BLINK = ('blink_l', 'blink_r', 'eyeBlinkLeft', 'eyeBlinkRight')
body = bpy.data.objects['Human']
lash = next(o for o in bpy.data.objects if o.type == 'MESH' and 'eyelash' in o.name)
eye = next(o for o in bpy.data.objects if o.type == 'MESH' and 'high-poly' in o.name)

bkeys = body.data.shape_keys.key_blocks
lkeys = lash.data.shape_keys.key_blocks
polys = [tuple(p.vertices) for p in body.data.polygons]
rest = [d.co.copy() for d in bkeys[0].data]
to_body = body.matrix_world.inverted() @ lash.matrix_world
from_body = lash.matrix_world.inverted() @ body.matrix_world

# eye centres (body space), one per side
eye_pts = [body.matrix_world.inverted() @ (eye.matrix_world @ v.co) for v in eye.data.vertices]
centres = {}
for side in (-1, 1):
    pts = [p for p in eye_pts if (p.x > 0) == (side > 0)]
    centres[side] = sum(pts, Vector()) / len(pts)


# Lid margins: the body's boundary edges around each eye opening. A lash's root is the nearest point
# on them (the nearest point on the whole body would put the root of a lash tip up on the lid skin).
counts = {}
for p in body.data.polygons:
    for k in p.edge_keys:
        counts[k] = counts.get(k, 0) + 1
margin = [k for k, c in counts.items() if c == 1
          and min((rest[k[0]] - centres[s]).length for s in (-1, 1)) < 0.025]
print('MARGIN EDGES', len(margin))


def nearest_on_margin(p):
    best = None
    for a, b in margin:
        ab = rest[b] - rest[a]
        t = max(0.0, min(1.0, (p - rest[a]).dot(ab) / max(ab.length_squared, 1e-12)))
        q = rest[a] + ab * t
        dist = (p - q).length
        if best is None or dist < best[0]:
            best = (dist, a, b, t, q)
    return best[1:]


lash_rest = [to_body @ d.co for d in lkeys[0].data]
roots = [nearest_on_margin(p) for p in lash_rest]   # (a, b, t, point)

for name in BLINK:
    if name not in lkeys or name not in bkeys:
        continue
    moved = [d.co.copy() for d in bkeys[name].data]
    kb = lkeys[name]
    # which eye closes: the side where the body key moves the most
    shift = [(m - r).length * (1 if r.x > 0 else -1) for m, r in zip(moved, rest)]
    side = 1 if sum(v for v in shift if v > 0) > -sum(v for v in shift if v < 0) else -1
    n = 0
    for i, d in enumerate(kb.data):
        ea, eb, t, loc = roots[i]
        if (loc.x > 0) != (side > 0):
            continue
        root = moved[ea] + (moved[eb] - moved[ea]) * t
        off = lash_rest[i] - loc
        c = centres[1 if loc.x > 0 else -1]
        if loc.z > c.z:   # upper lash
            # Closed, the upper lashes go inside the closed lid (CLOSED_SINK of the way from the margin to the eye's
            # centre). Every way of laying the flat MPFB cards on the closed lid showed somewhere: mirrored or folded
            # onto the margin they lie flat, and the closed margin spreads out front to back, so from above the
            # squeezed strands read as a striped band; hung down, they read as a curtain with a gap from the front.
            off = (c - root) * CLOSED_SINK
        else:            # lower lash: mostly tucked under the closed lid
            off = off * LOWER_KEEP
        d.co = from_body @ (root + off)
        n += 1
    print('LASH BLINK', name, n)

lash.data.update()
bpy.ops.wm.save_as_mainfile(filepath=out)
