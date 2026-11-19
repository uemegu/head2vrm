"""Keep the eyebrows still, and the painted lashes unstretched, when the eyes close.

blender --background FACE.blend --python blink_brow.py -- OUT.blend

Why: the blink keys come from the MakeHuman base, whose upper lid region reaches 14 mm above the eye. An input face
with low brows has its brow hair there, so closing the eye pulled the brow's lower half 3-6 mm down and its painted hairs ran down
the closed lid as streaks that looked like eyebrows.

In the lid-closing keys (eyeBlink*, blink_l/r, eyeSquint*), the vertices on brow hair (the skin texture under BROW_DARK of the
skin round the eye, and BROW_ABOVE mm or more above the upper lid's edge) do not move; under the brow's lower edge the motion comes
back evenly (linearly) over the skin between the brow's lower edge and the lid's edge, so the lid skin takes the stretch spread out
(a 3 mm ramp under the brow gathered it into a crease line there). The lid's edge
moves as before, so the eye still shuts.

The painted lashes (LASH=1): the input texture may paint the upper lashes on the skin above the lid's edge. The blink keys move that edge furthest
and the skin above it less, so the painted band was stretched down the closed lid into vertical streaks. In the blink keys
(eyeBlink*, blink_l/r) each vertex of the band (from the lid's edge up LASH_MAX mm, under the brow) stays where the key puts it
on the closed lid but no further from its lid edge vertex than when open; above it the motion goes back to the key's own (after the brow rule above) over LASH_RAMP mm, so the
plain skin there takes the stretch. The open eye is unchanged (only the keys change).
"""
import os
import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index('--') + 1]
E = lambda k, d: float(os.environ.get(k, d))
BROW_DARK, BROW_ABOVE, RAMP, BIN = E('BROW_DARK', 0.7), E('BROW_ABOVE', 3) / 1000, E('RAMP', 3) / 1000, E('BIN', 1.5) / 1000
BROW_BAND = E('BROW_BAND', 16) / 1000   # how far above the lid's edge the brow is looked for (the painted scalp lies above)
RAMP_FRAC = E('RAMP_FRAC', 0.5)
LASH, LASH_MAX, LASH_RAMP = E('LASH', 1), E('LASH_MAX', 5) / 1000, E('LASH_RAMP', 2) / 1000
LASH_SIDE = E('LASH_SIDE', 3) / 1000
LASH_COL = E('LASH_COL', 2) / 1000
LASH_BACK = E('LASH_BACK', 12) / 1000
LASH_GAP = E('LASH_GAP', 8) / 1000

o = bpy.data.objects['Human']; me = o.data; kb = me.shape_keys.key_blocks; n = len(me.vertices)
img = next(nd.image for nd in me.materials[0].node_tree.nodes if nd.type == 'TEX_IMAGE' and nd.image)
W, H = img.size; px = np.empty(W * H * 4, np.float32); img.pixels.foreach_get(px); lum = px.reshape(H, W, 4)[..., :3].mean(2)
uv = np.zeros((n, 2)); cnt = np.zeros(n); uvd = me.uv_layers.active.data
for l in me.loops:
    uv[l.vertex_index] += uvd[l.index].uv; cnt[l.vertex_index] += 1
uv /= np.maximum(cnt, 1)[:, None]
Lv = lum[np.clip((uv[:, 1] * H).astype(int), 0, H - 1), np.clip((uv[:, 0] * W).astype(int), 0, W - 1)]
base = np.array([v.co[:] for v in kb[0].data])
keys = [k for k in kb if k.name.startswith(('eyeBlink', 'blink_', 'eyeSquint'))]
if not keys:
    raise SystemExit('BLINK_BROW no lid keys')
disp = {k.name: np.array([v.co[:] for v in k.data]) - base for k in keys}

# the brow as an area: the face drawn flat from the front (orthographic, texture only) round each eye; brow = the dark pixels
# above the lid (closed up so the gaps between painted hairs count), its lower edge per column. A per-vertex texel test missed the
# brow's thin lower edge, where the hairs are sparse.
import tempfile
sc = bpy.context.scene
for ob in bpy.data.objects:
    if ob.type == 'MESH' and ob is not o:
        ob.hide_render = True
for k in kb:
    k.value = 0
cam = bpy.data.cameras.new('bb_cam'); cam.type = 'ORTHO'; SPAN = 0.06; cam.ortho_scale = SPAN
co_ = bpy.data.objects.new('bb_cam', cam); sc.collection.objects.link(co_); sc.camera = co_
co_.rotation_euler = (np.pi / 2, 0, 0)
sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'FLAT'; sc.display.shading.color_type = 'TEXTURE'
sc.view_settings.view_transform = 'Standard'; sc.render.film_transparent = True; sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGBA'; RES = 900; sc.render.resolution_x = sc.render.resolution_y = RES
mw = np.array(o.matrix_world); wb = (mw[:3, :3] @ base.T).T + mw[:3, 3]


def _filt(m, k, f):   # square max (dilate) / min (erode) filter of a bool image, k odd
    r = k // 2; p_ = np.pad(m, r, constant_values=(f is np.minimum))
    out_ = p_[r:-r or None, r:-r or None].copy()
    for a in range(-r, r + 1):
        out_ = f(out_, p_[r + a:p_.shape[0] - r + a, r:-r or None])
    o2 = out_.copy(); p2 = np.pad(out_, r, constant_values=(f is np.minimum))
    for a in range(-r, r + 1):
        o2 = f(o2, p2[r:-r or None, r + a:p2.shape[1] - r + a])
    return o2


def _largest(m, cols):   # the 8-connected area of a bool image with the most pixels in the given columns
    from collections import deque
    lab_ = np.zeros(m.shape, np.int32); best, best_n, cur = 0, 0, 0
    H_, W_ = m.shape
    for y0, x0 in zip(*np.nonzero(m)):
        if lab_[y0, x0]:
            continue
        cur += 1; q = deque([(y0, x0)]); lab_[y0, x0] = cur; c_ = 0
        while q:
            y, x = q.popleft(); c_ += int(cols[x])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy_, xx_ = y + dy, x + dx
                    if 0 <= yy_ < H_ and 0 <= xx_ < W_ and m[yy_, xx_] and not lab_[yy_, xx_]:
                        lab_[yy_, xx_] = cur; q.append((yy_, xx_))
        if c_ > best_n:
            best, best_n = cur, c_
    return lab_ == best if best else np.zeros_like(m)


def _read(f):
    im_ = bpy.data.images.load(f); w_, h_ = im_.size; a_ = np.empty(w_ * h_ * 4, np.float32); im_.pixels.foreach_get(a_)
    bpy.data.images.remove(im_)
    return a_.reshape(h_, w_, 4)[::-1] * 255   # top row first, sRGB 0-255 (Standard view transform), alpha last


scale = np.ones(n)
follow = np.zeros(n); edge_of = np.arange(n)   # the painted lash band: how much each vertex follows its lid edge vertex, and which
for side in (-1, 1):
    bk = [k for k in ('eyeBlinkLeft', 'eyeBlinkRight', 'blink_l', 'blink_r') if k in disp]
    dm = np.max([np.linalg.norm(disp[k], axis=1) for k in bk], 0) * (np.sign(base[:, 0]) == side)
    mv = dm > 3e-4
    if mv.sum() < 20:
        continue
    lid = mv & (dm > 0.8 * dm[mv].max())                     # the upper lid's edge: the vertices that travel furthest
    cx, cz = wb[lid, 0].mean(), wb[lid, 2].mean() + 0.008
    co_.location = (cx, wb[:, 1].min() - 0.3, cz)
    f = os.path.join(tempfile.mkdtemp(), 'bb.png'); sc.render.filepath = f
    bpy.ops.render.render(write_still=True)
    im4 = _read(f); im = im4[..., :3]; L = im.mean(2); face = im4[..., 3] > 128
    to_px = lambda p: (np.round((p[:, 0] - cx) / SPAN * RES + RES / 2).astype(int), np.round(RES / 2 - (p[:, 2] - cz) / SPAN * RES).astype(int))
    lx, ly = to_px(wb[lid])
    # the lid's edge as a line over the image's columns (the lid vertices' highest point per column, filled in between)
    lid_y = np.full(RES, np.nan)
    for x_, y_ in zip(lx, ly):
        if 0 <= x_ < RES:
            lid_y[x_] = np.nanmin([lid_y[x_], y_])
    ok_ = ~np.isnan(lid_y); lid_y = np.interp(np.arange(RES), np.nonzero(ok_)[0], lid_y[ok_])
    yy = np.arange(RES)[:, None]
    above = (yy < lid_y[None, :] - BROW_ABOVE / SPAN * RES) & (yy > lid_y[None, :] - BROW_BAND / SPAN * RES)
    skin = np.median(L[(yy < lid_y[None, :]) & (yy > lid_y[None, :] - 0.004 / SPAN * RES) & (L > 20)])
    dark = (L < BROW_DARK * skin) & above & face
    k_ = max(3, int(0.0015 / SPAN * RES) | 1)
    dark = _filt(_filt(dark, k_, np.maximum), k_, np.minimum) & above & face
    over_eye = np.zeros(RES, bool); over_eye[np.clip(lx.min(), 0, RES - 1):np.clip(lx.max(), 0, RES - 1) + 1] = True
    brow_m = _largest(dark, over_eye)
    if not brow_m.any():
        print('BLINK_BROW side %+d: no brow found above the lid' % side)
    cols = brow_m.any(0); z_b_px = np.where(cols, RES - 1 - np.argmax(brow_m[::-1], 0), -1)   # lowest brow pixel per column
    # smoothed along the brow (a running median over 2 mm): the mask's ragged lower edge put kinks into the closed lid
    w_ = max(1, int(0.001 / SPAN * RES)); zb_s = z_b_px.astype(float); zb_s[~cols] = np.nan
    zb_s = np.array([np.nanmedian(zb_s[max(0, i - w_):i + w_ + 1]) if cols[i] else np.nan for i in range(RES)]) if cols.any() else zb_s
    z_b_px = np.where(cols, np.round(zb_s), -1).astype(int)
    chk = im.copy(); chk[brow_m, 0] = 255   # the brow found in red, the lid edge in green (a check image)
    for x_ in range(RES):
        chk[int(np.clip(lid_y[x_], 0, RES - 1)), x_] = (0, 255, 0)
    ci = bpy.data.images.new('bb_chk', RES, RES); ci.pixels.foreach_set(np.dstack([chk[::-1] / 255, np.ones((RES, RES, 1))]).astype(np.float32).ravel())
    ci.filepath_raw = os.path.join(os.path.dirname(os.path.abspath(out)), 'blink_brow_%s.png' % ('r' if side < 0 else 'l')); ci.file_format = 'PNG'; ci.save()
    bpy.data.images.remove(ci)
    vx, vy = to_px(wb)
    inn = mv & (vx >= 0) & (vx < RES)
    idx = np.nonzero(inn)[0]
    if LASH:
        # the painted lash band: from the lid's edge (a quadratic through the lid edge vertices: the per-column highest vertex
        # zigzagged up into the lashes) up LASH_MAX mm, stopping 1 mm under the brow and at 0.6 of the gap to it. Found by height,
        # not darkness: light, thin painted lashes did not pass a darkness test (a 0.8 mm band where they reach about 4 mm)
        pc_ = np.polyfit(lx, ly, 2); xs_ = np.arange(RES)
        # the band reaches LASH_SIDE mm past the lid edge vertices on each side (the outer lashes run past the last of them)
        m_ = LASH_SIDE / SPAN * RES
        edge = np.polyval(pc_, np.clip(xs_, lx.min() - m_, lx.max() + m_))
        okt = (xs_ >= lx.min() - m_) & (xs_ <= lx.max() + m_)
        top = edge - LASH_MAX / SPAN * RES
        hb = z_b_px >= 0
        top[hb] = np.maximum(top[hb], np.maximum(z_b_px[hb] + 0.001 / SPAN * RES, edge[hb] - 0.6 * (edge[hb] - z_b_px[hb])))
        # only where there is room for the stretch between the band and the brow (LASH_GAP mm or more from the lid's edge to the
        # brow): with a low brow, shrinking the band pushed the stretch up into the brow's edge and its hairs streaked the closed
        # lid again (the man's short painted lashes had no streaks to remove)
        okt &= ~hb | ((edge - z_b_px) * SPAN / RES >= LASH_GAP)
        tm = np.where(okt, top, np.nan)
        if okt.sum() > 10:
            lidv = np.nonzero(lid)[0]
            for i in idx:
                x_ = vx[i]
                if np.isnan(tm[x_]) or vy[i] >= edge[x_] - 0.0005 / SPAN * RES:
                    continue
                r_ = max(1.0, LASH_RAMP / SPAN * RES)
                w = float(np.clip(1 - (tm[x_] - vy[i]) / r_, 0, 1)) if vy[i] < tm[x_] else 1.0
                # its lid edge vertex: the nearest in 3D among those within LASH_COL mm of its column (the nearest of all was at times
                # another part of the lid, and pulling to it threw lid pieces out; the nearest by column alone left steps)
                cand = lidv[np.abs(lx - x_) <= LASH_COL / SPAN * RES]
                if w <= 0 or not len(cand):
                    continue
                e_i = cand[np.argmin(np.linalg.norm(base[cand] - base[i], axis=1))]
                # only the lid's outer skin (within LASH_BACK mm of its edge vertex in depth): the eye socket's inner vertices lie about 20 mm behind it and, pulled to the closed lid's
                # edge, came 30 mm forward through it as spikes (-y is the front)
                if wb[i, 1] - wb[e_i, 1] < LASH_BACK:
                    follow[i] = w; edge_of[i] = e_i
            for x_ in np.nonzero(okt)[0]:
                chk[int(np.clip(edge[x_], 0, RES - 1)), x_] = (255, 255, 0)
            chk[np.clip(np.nan_to_num(tm, nan=0).astype(int), 0, RES - 1)[okt], np.nonzero(okt)[0]] = (0, 128, 255)
            ci = bpy.data.images.new('bb_chk2', RES, RES); ci.pixels.foreach_set(np.dstack([chk[::-1] / 255, np.ones((RES, RES, 1))]).astype(np.float32).ravel())
            ci.filepath_raw = os.path.join(os.path.dirname(os.path.abspath(out)), 'blink_brow_%s.png' % ('r' if side < 0 else 'l')); ci.file_format = 'PNG'; ci.save()
            bpy.data.images.remove(ci)
            print('BLINK_BROW side %+d: painted lash band over %d columns, %.1f mm high (median); %d vertices move with the lid edge'
                  % (side, okt.sum(), np.nanmedian(edge[okt] - top[okt]) * SPAN / RES * 1000, ((follow > 0) & (np.sign(base[:, 0]) == side)).sum()))
    zb = z_b_px[vx[idx]]; has = zb >= 0
    gap_px = zb - lid_y[vx[idx]]                                   # negative: brow pixels' bottom is above the lid (px grow down)
    # 0 at the brow's lower edge, back to 1 by RAMP_FRAC of the way to the lid's edge (smoothstep): the lower part of the lid keeps
    # its full motion, so the painted lid line moves with the lid (easing all the way down smeared it over the closed lid)
    t = np.clip((vy[idx] - zb) / np.maximum(-gap_px * RAMP_FRAC, 1), 0, 1); s_ = t * t * (3 - 2 * t)
    s_[~has] = 1
    s_[vy[idx] >= lid_y[vx[idx]] - 0.001 / SPAN * RES] = 1          # the lid's edge and below: as before
    scale[idx] = np.minimum(scale[idx], s_)
    print('BLINK_BROW side %+d: brow %d px, columns %d; moving vertices held still %d, eased %d'
          % (side, brow_m.sum(), cols.sum(), (scale[idx] < 0.02).sum(), ((scale[idx] >= 0.02) & (scale[idx] < 0.98)).sum()))
bpy.data.objects.remove(co_)

for k in keys:
    d = disp[k.name] * scale[:, None]
    if k.name.startswith(('eyeBlink', 'blink_')):
        # on the closed lid, each band vertex keeps the key's own place on the lid's surface but is pulled along the line to its
        # lid edge vertex until it is as far from it as when open: the band is not stretched, and the vertices stay on the lid
        # (moving them as one piece with the edge folded the lid over the eyeball into faceted steps)
        fi = np.nonzero(follow > 0)[0]
        if len(fi):
            e_ = edge_of[fi]; ce = base[e_] + disp[k.name][e_] * scale[e_, None]; cv_ = base[fi] + d[fi]
            # the stretch as seen from the front: the height above the lid edge vertex (z), open against closed (the 3D distance
            # missed it, the lid closing round the eyeball)
            r_open = np.maximum(base[fi, 2] - base[e_, 2], 0); r_cl = np.maximum(cv_[:, 2] - ce[:, 2], 1e-6)
            tgt = ce + (cv_ - ce) * np.minimum(1, r_open / r_cl)[:, None]
            d[fi] = d[fi] * (1 - follow[fi, None]) + (tgt - base[fi]) * follow[fi, None]
    co = (base + d).ravel()
    k.data.foreach_set('co', co)
me.update()
print('BLINK_BROW keys', ', '.join(k.name for k in keys))
bpy.ops.wm.save_as_mainfile(filepath=out)
