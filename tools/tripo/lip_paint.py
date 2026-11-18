"""The lips painted, not copied: a smooth colour field measured on Tripo's lips, its fine lines kept, no painted gloss or seam
shadow, and the lips' inner sides the colour of wet lip.   any face, after mouth_arap.py (it reads the approved mouth lines, the seam
curve and the lid-fit camera)
python lip_paint.py OUT_DIR SKIN_IN SKIN_OUT

Why (the lips are painted here so they still look right when the mouth opens): Tripo
painted a closed mouth on one surface, its seam's dark line, its light and a white gloss streak included. Copied onto our lips it gave
a pale band above the seam and a ragged seam, and with the mouth open the lips' inner sides showed skin colour and white.
- Where (in the lid-fit camera): the lips lie between the approved outer lines (splines through their points); the seam is
  mouth_arap.py's curve. Each point has lip coordinates: u along the mouth (0 left corner, 1 right), v across a lip (0 at the seam,
  1 at its outer line).
- Colour: Tripo's lips (lidfit/warp/v0_tripo.png, unlit) sampled at many (u, v) of each lip (a 3x3 median in Lab), leaving out the
  gloss (lighter than LIGHT over the lip's median and paler) and the seam band (v < V_SEAM) and the border (v > V_EDGE); per lip and
  channel, a least-squares surface (degree DU in u, DV in v), refitted without the worst residuals. No line is drawn on the seam:
  the seam's shadow is left to the light (a drawn seam line is not needed; the shading makes it).
- Fine lines: Tripo's lightness minus its blur (sigma DETAIL_S px), times DETAIL, except on the gloss and the seam band.
- The border: the lip colour fades into the skin over EDGE_W px across the outer line (Tripo's redness falls over about that).
- The inner sides: texels the camera does not see, within INNER mm of the seam rows, take the seam-side lip colour turning to wet lip
  (Lab: lightness x WET_L, a* + WET_A) by WET_MM, fading out to the texture as it was by INNER.
Writes SKIN_OUT and OUT_DIR/mouthfit/lip_paint_check.jpg (Tripo, its field, the field with lines, as seen from the camera).
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

O, src, dst = sys.argv[1:4]
E = lambda k, d: float(os.environ.get(k, d))
V_SEAM, V_EDGE, LIGHT, DU, DV = E('V_SEAM', 0.2), E('V_EDGE', 0.8), E('LIGHT', 10), int(E('DU', 4)), int(E('DV', 2))
DETAIL, DETAIL_S, EDGE_W = E('DETAIL', 1.0), E('DETAIL_S', 2.5), E('EDGE_W', 8)
INNER, WET_MM, WET_L, WET_A = E('INNER', 6) / 1000, E('WET_MM', 1.5) / 1000, E('WET_L', 0.82), E('WET_A', 6)

cam = json.load(open(f'{O}/lidfit/warp/v0_cam.json'))
Mw = np.array(cam['cam_world']); Rc, Cc = Mw[:3, :3], Mw[:3, 3]
RES = cam['res'][1]; F = (RES / 2) / np.tan(cam['fov'] / 2)


def to_cam(P):
    q = (P - Cc) @ Rc; d = -q[..., 2]
    return np.stack([RES / 2 + F * q[..., 0] / d, RES / 2 - F * q[..., 1] / d, d], -1)


OUTER_U = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
OUTER_L = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
pts = {int(r.split()[0]): np.array([float(v) for v in r.split()[1:]]) for r in open(f'{O}/mouthfit/check_lines.txt') if r.strip()}
sc = np.load(f'{O}/mouthfit/seam_curve.npz')
x0, x1 = pts[61][0], pts[291][0]


def line_y(ids):
    Q = np.array([pts[i] for i in ids]); t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    d = CubicSpline(t, Q)(np.linspace(0, t[-1], 3000)); o = np.argsort(d[:, 0])
    return lambda x: np.interp(x, d[o, 0], d[o, 1])


up_y, lo_y = line_y(OUTER_U), line_y(OUTER_L)
seam_y = lambda x: np.interp(x, sc['x'], sc['y'])


def lip_uv(x, y):
    """u, v, which lip (+1 upper, -1 lower, 0 outside the corners), signed px inside the outer line"""
    u = (x - x0) / (x1 - x0); s = seam_y(x); a, b = up_y(x), lo_y(x)
    upper = y < s
    v = np.where(upper, (s - y) / np.maximum(s - a, 1e-3), (y - s) / np.maximum(b - s, 1e-3))
    inside = np.where(upper, y - a, b - y)
    lip = np.where((u > 0) & (u < 1), np.where(upper, 1, -1), 0)
    return u, v, lip, inside


# --- the colour field, measured on Tripo
tri = cv2.imread(f'{O}/lidfit/warp/v0_tripo.png')[..., :3]
lab = cv2.cvtColor(tri, cv2.COLOR_BGR2LAB).astype(np.float32)
lab_med = cv2.medianBlur(cv2.cvtColor(tri, cv2.COLOR_BGR2LAB), 3).astype(np.float32)
H, W = tri.shape[:2]
gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
bx0, bx1 = int(x0) - 4, int(x1) + 5
by0, by1 = int(min(up_y(np.linspace(x0, x1, 200)).min(), seam_y(x0))) - 4, int(lo_y(np.linspace(x0, x1, 200)).max()) + 5
U_, V_, LIP_, IN_ = lip_uv(gx[by0:by1, bx0:bx1] + 0.5, gy[by0:by1, bx0:bx1] + 0.5)
L_ = lab_med[by0:by1, bx0:bx1]
chroma = np.hypot(L_[..., 1] - 128, L_[..., 2] - 128)
gloss = np.zeros(U_.shape, bool)
fields = {}
for side in (1, -1):
    m = (LIP_ == side) & (V_ > 0.05) & (V_ < 0.95) & (U_ > 0.03) & (U_ < 0.97)
    Lm, Cm = np.median(L_[..., 0][m]), np.median(chroma[m])
    # the gloss: lighter than its surroundings (lightness minus its 6 px blur over LIGHT) or than the lip (over 2 LIGHT); the
    # first test alone (lighter than the lip's median and paler) found only 220 px of the lower lip's streak
    hp = L_[..., 0] - cv2.GaussianBlur(L_[..., 0], (0, 0), 6)
    g = m & ((hp > LIGHT * 255 / 100 * 0.5) | (L_[..., 0] > Lm + 2 * LIGHT * 255 / 100))
    gloss |= cv2.dilate(g.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    s = m & ~gloss & (V_ > V_SEAM) & (V_ < V_EDGE)
    u, v, c = U_[s], V_[s], L_[s]
    A = np.stack([u ** i * v ** j for i in range(DU + 1) for j in range(DV + 1)], 1)
    keep = np.ones(len(u), bool); coef = []
    for ch in range(3):
        k_ = keep.copy()
        for _ in range(3):
            cf = np.linalg.lstsq(A[k_], c[k_, ch], rcond=None)[0]
            r = np.abs(A @ cf - c[:, ch]); k_ = r < max(np.percentile(r[k_], 85), 1.0)
        coef.append(cf)
    fields[side] = np.array(coef)
    print('LIP_PAINT %s lip: %d samples, gloss px %d, median Lab %s' % ('upper' if side > 0 else 'lower', s.sum(), g.sum(),
          np.round(np.median(c, 0) * [100 / 255, 1, 1] - [0, 128, 128], 1)))


def base_field(u, v, side):
    u = np.clip(u, 0, 1); v = np.clip(v, V_SEAM, V_EDGE)   # held flat beyond the measured band: no seam shadow, no border
    A = np.stack([u ** i * v ** j for i in range(DU + 1) for j in range(DV + 1)], -1)
    return A @ fields[side].T


# ACROSS (without it the lips looked flat in colour): the viewer's B set shows the lips unlit, so the dark by the seam and the
# border's fade, left out above for the light to make, never showed (the lower lip's L* spread 4.3 against Tripo's 16.4).
# Tripo's own profile across each lip is put back: per lip, the median of Tripo minus the field above in bins of v (0 seam .. 1
# outer line), over the middle of the mouth (u 0.15-0.85, the gloss left out), added at every u. One curve across, not a surface:
# a degree-4 surface in both u and v ran off at the corners (L* 121 at the upper lip's right corner, -6 at the lower lip's left).
ACROSS = E('ACROSS', 0)
prof_v = np.linspace(0, 1, 41); profs = {1: np.zeros((41, 3)), -1: np.zeros((41, 3))}
if ACROSS:
    for side in (1, -1):
        m = (LIP_ == side) & ~gloss & (U_ > 0.15) & (U_ < 0.85) & (V_ >= 0) & (V_ <= 1)
        r = L_[m] - base_field(U_[m], V_[m], side)
        b_ = np.clip((V_[m] * 40).round().astype(int), 0, 40)
        p = np.array([np.median(r[b_ == k], 0) if (b_ == k).sum() > 8 else np.full(3, np.nan) for k in range(41)])
        for ch in range(3):
            ok_ = ~np.isnan(p[:, ch]); p[:, ch] = np.interp(prof_v, prof_v[ok_], p[ok_, ch])
        profs[side] = gaussian_filter(p, (0.8, 0), mode='nearest')
        print('LIP_PAINT across %s lip: L* added at v 0 / .1 / .5 / .9 / 1: %s' % ('upper' if side > 0 else 'lower',
              ' / '.join('%+.1f' % (profs[side][k, 0] * 100 / 255) for k in (0, 4, 20, 36, 40))))


def field(u, v, side, across=True):
    c = base_field(u, v, side)
    if across: c = c + np.stack([np.interp(np.clip(v, 0, 1), prof_v, profs[side][:, ch]) for ch in range(3)], -1)
    return c


Lb = cv2.GaussianBlur(lab[..., 0], (0, 0), DETAIL_S)
detail = (lab[..., 0] - Lb)
dmask = np.ones((H, W), np.float32)
dm = dmask[by0:by1, bx0:bx1]; dm[gloss] = 0
seam_px = np.abs((gy[by0:by1, bx0:bx1] + 0.5) - seam_y(gx[by0:by1, bx0:bx1] + 0.5)) < 3
dm[seam_px] = 0
dmask = cv2.GaussianBlur(dmask, (0, 0), 1.5)
detail *= dmask * DETAIL
# the sparkle (SPARKLE): a reference face's lips glint from the fine light streaks Tripo painted on its lower lip, kept
# in its texture. Another test head's Tripo has the same streaks in the same place, 2.3 times as strong (lightness above its 6 px blur, 99th
# percentile: reference 4.2, test head 9.9) and bluish white (b 4 against the lip's 11; the reference's keep the lip's b). Inside the gloss found above
# they come back at SPARKLE of their strength (0.43 matched the reference in the texture but vanished in the viewer: the test head's texture holds 0.25 mm a texel, the reference's 0.07 mm; 1.0 kept), on the painted lip's own colour, a little less red (SPARKLE_A per unit of lightness)
SPARKLE, SPARKLE_A = E('SPARKLE_TRIPO', 0), E('SPARKLE_A', 1.0)   # off: drawn lines below instead
sparkle = np.zeros((H, W), np.float32)
hp_full = lab[..., 0] - cv2.GaussianBlur(lab[..., 0], (0, 0), 6)
sm = np.zeros((H, W), np.float32); sm_ = sm[by0:by1, bx0:bx1]; sm_[gloss & ~seam_px] = 1
sm = cv2.GaussianBlur(sm, (0, 0), 1.0)
sparkle = np.maximum(hp_full, 0) * sm * SPARKLE
if SPARKLE: print('LIP_PAINT sparkle: %d px, lightness added p99 %.1f max %.1f' % ((sparkle > 1).sum(), np.percentile(sparkle[sparkle > 0], 99) * 100 / 255 if (sparkle > 0).any() else 0, sparkle.max() * 100 / 255))


def lip_colour(x, y):
    """Lab (OpenCV scale) of the painted lip and its weight (0..1) at screen points"""
    u, v, side, inside = lip_uv(x, y)
    col = np.where((side > 0)[:, None], field(u, v, 1), field(u, v, -1))
    xi = np.clip(x.astype(int), 0, W - 1); yi = np.clip(y.astype(int), 0, H - 1)
    col[:, 0] += detail[yi, xi] + sparkle[yi, xi]
    col[:, 1] -= SPARKLE_A * sparkle[yi, xi] * 100 / 255
    wgt = np.clip(inside / EDGE_W + 0.5, 0, 1) * (side != 0)
    corner = np.clip(np.minimum(u, 1 - u) * (x1 - x0) / (EDGE_W * 0.5), 0, 1)   # the corners fade too
    return col, wgt * corner


# the check, as seen from the camera
yy, xx = gy[by0:by1, bx0:bx1].ravel() + 0.5, gx[by0:by1, bx0:bx1].ravel() + 0.5
cl, wl = lip_colour(xx, yy)
flat = cl.copy(); flat[:, 0] -= detail[yy.astype(int), xx.astype(int)] + sparkle[yy.astype(int), xx.astype(int)]; flat[:, 1] += SPARKLE_A * sparkle[yy.astype(int), xx.astype(int)] * 100 / 255
crop = tri[by0:by1, bx0:bx1].astype(np.float32)
rows = [crop]
for cc in (flat, cl):
    rgb = cv2.cvtColor(np.clip(cc, 0, 255).astype(np.uint8).reshape(by1 - by0, bx1 - bx0, 3), cv2.COLOR_LAB2BGR).astype(np.float32)
    w_ = wl.reshape(by1 - by0, bx1 - bx0, 1)
    rows.append(crop * (1 - w_) + rgb * w_)
chk = np.vstack(rows).astype(np.uint8)
cv2.imwrite(f'{O}/mouthfit/lip_paint_check.jpg', cv2.resize(chk, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))

# --- onto the texture
t = np.load(f'{O}/texels.npz'); pos, have = t['pos'], t['have']
u_ = np.load(f'{O}/uvmesh.npz'); Vm, tv = u_['co'], u_['tv']
img = np.asarray(Image.open(src).convert('RGB'), np.float32) / 255
ys, xs = np.nonzero(have)
Q = to_cam(pos[ys, xs])
near = (Q[:, 0] > bx0 - 30) & (Q[:, 0] < bx1 + 30) & (Q[:, 1] > by0 - 30) & (Q[:, 1] < by1 + 30) & (Q[:, 2] > 0)
ys, xs, Q = ys[near], xs[near], Q[near]
# what the camera sees: a z-buffer of our own skin over the box
S = to_cam(Vm[tv]); box = (bx0 - 30, by0 - 30, bx1 + 30, by1 + 30)
inb = (S[..., 0].max(1) >= box[0]) & (S[..., 0].min(1) <= box[2]) & (S[..., 1].max(1) >= box[1]) & (S[..., 1].min(1) <= box[3])
Z = np.full((box[3] - box[1], box[2] - box[0]), np.inf)
for s in S[inb]:
    if (s[:, 2] <= 0).any(): continue
    ax0, ay0 = max(int(np.floor(s[:, 0].min())), box[0]), max(int(np.floor(s[:, 1].min())), box[1])
    ax1, ay1 = min(int(np.ceil(s[:, 0].max())) + 1, box[2]), min(int(np.ceil(s[:, 1].max())) + 1, box[3])
    if ax0 >= ax1 or ay0 >= ay1: continue
    px_, py_ = np.meshgrid(np.arange(ax0, ax1) + 0.5, np.arange(ay0, ay1) + 0.5)
    (ax, ay), (bx_, by_), (cx, cy) = s[0, :2], s[1, :2], s[2, :2]
    den = (by_ - cy) * (ax - cx) + (cx - bx_) * (ay - cy)
    if abs(den) < 1e-12: continue
    l0 = ((by_ - cy) * (px_ - cx) + (cx - bx_) * (py_ - cy)) / den; l1 = ((cy - ay) * (px_ - cx) + (ax - cx) * (py_ - cy)) / den
    l2 = 1 - l0 - l1; ins = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
    z = 1 / (l0 / s[0, 2] + l1 / s[1, 2] + l2 / s[2, 2])
    sub = Z[ay0 - box[1]:ay1 - box[1], ax0 - box[0]:ax1 - box[0]]; win = ins & (z < sub); sub[win] = z[win]
ix = np.clip(Q[:, 0].astype(int) - box[0], 0, Z.shape[1] - 1); iy = np.clip(Q[:, 1].astype(int) - box[1], 0, Z.shape[0] - 1)
visible = Q[:, 2] <= Z[iy, ix] + 0.0004
col, wgt = lip_colour(Q[:, 0], Q[:, 1])
wgt *= visible
Wt = np.zeros(have.shape, np.float32); Pc = np.zeros(img.shape, np.float32)
rgb = cv2.cvtColor(np.clip(col, 0, 255).astype(np.uint8)[None], cv2.COLOR_LAB2RGB)[0].astype(np.float32) / 255
Wt[ys, xs] = wgt; Pc[ys, xs] = rgb
# the inner sides: hidden texels near the seam rows
rows_ = np.array(list(sc['rowU']) + list(sc['rowL']))
tree = cKDTree(Vm[rows_])
hid = ~visible
d_, j_ = tree.query(pos[ys[hid], xs[hid]])
ok = d_ < INNER
hy, hx = ys[hid][ok], xs[hid][ok]; d_ = d_[ok]
# the seam-side colour of the lip that row belongs to, at that row vertex's u
rv = Vm[rows_[j_[ok]]]; rq = to_cam(rv)
side = np.where(np.isin(rows_[j_[ok]], sc['rowU']), 1, -1)
ur = np.clip((rq[:, 0] - x0) / (x1 - x0), 0, 1)
# the inner sides (seen with the mouth open) take the lip without the seam's dark (ACROSS adds it to the outside only)
seam_col = np.where((side > 0)[:, None], field(ur, np.zeros_like(ur), 1, False), field(ur, np.zeros_like(ur), -1, False))
wet = seam_col.copy(); wet[:, 0] *= WET_L; wet[:, 1] += WET_A
a = np.clip(d_ / WET_MM, 0, 1)[:, None]
inner_col = seam_col * (1 - a) + wet * a
inner_w = 1 - np.clip((d_ - (INNER - 1.5e-3)) / 1.5e-3, 0, 1)
Pc[hy, hx] = cv2.cvtColor(np.clip(inner_col, 0, 255).astype(np.uint8)[None], cv2.COLOR_LAB2RGB)[0].astype(np.float32) / 255
Wt[hy, hx] = np.maximum(Wt[hy, hx], inner_w)
print('LIP_PAINT texels: visible lip %d, inner sides %d' % ((wgt > 0.5).sum(), (inner_w > 0.5).sum()))
Hv = have.astype(np.float32)
Ws = gaussian_filter(Wt, 1.5) / np.maximum(gaussian_filter(Hv, 1.5), 1e-6) * Hv * (Wt > 0)
Pc_s = gaussian_filter(Pc * Wt[..., None], (1.0, 1.0, 0)) / np.maximum(gaussian_filter(Wt, 1.0), 1e-6)[..., None]
out = img * (1 - Ws[..., None]) + Pc_s * Ws[..., None]
# gloss lines drawn by us (GLOSS_LINES; the lips need not be Tripo's texture, so the gloss is drawn rather than copied).
# A reference face's lips glint from short fine light dashes along the lip's vertical lines over the upper half of the lower lip; Tripo's own
# (on another test head) came out as soft blobs, being 0.1-0.2 mm wide on a texture of 0.25 mm a texel and blurred twice on the way. These are
# evaluated at each visible texel from the lip's own (u, v) frame, unblurred: GL_N dashes, centres about u 0.5 (sd GL_USD) and
# v GL_V (sd 0.08) below the seam, GL_LEN mm long (0.4-1.0 x), GL_W mm wide (a Gaussian), leaning out from the middle as the lip's
# lines fan, each GL_L x (0.4-1) of lightness (CIE) added on the lip's own colour, a little less red (SPARKLE_A per unit)
if E('GLOSS_LINES', 1):
    MMPX = np.linalg.norm(Cc - np.array(cam['centre'])) / F * 1000
    rng = np.random.default_rng(int(E('GL_SEED', 8)))
    n_ = int(E('GL_N', 70))
    gu = np.clip(rng.normal(0.5, E('GL_USD', 0.12), n_), 0.22, 0.78); gv = np.clip(rng.normal(E('GL_V', 0.33), 0.08, n_), 0.12, 0.6)
    glen = E('GL_LEN', 1.4) * rng.uniform(0.4, 1.0, n_) / MMPX; gb = E('GL_L', 9) * rng.uniform(0.4, 1.0, n_)
    gw = E('GL_W', 0.12) / MMPX

    def gloss_lines(x, y):
        u, v, side, inside = lip_uv(x, y)
        hpx = lo_y(x) - seam_y(x)   # the lower lip's height here, px
        out_ = np.zeros(len(x), np.float32)
        for k in range(n_):
            dy = (v - gv[k]) * hpx; dx = (u - gu[k]) * (x1 - x0) - dy * 0.8 * (gu[k] - 0.5)
            along = np.clip(1 - np.abs(dy) / (glen[k] / 2), 0, 1) ** 0.5
            out_ = np.maximum(out_, gb[k] * np.exp(-dx ** 2 / (2 * gw ** 2)) * along)
        return out_ * (side < 0) * (inside > 2)
    if os.environ.get('GLOSS_FROM'):   # another face's own lines (gloss_from.py; by default the reference face's, data/gloss_aoi.npz), lower lip, at the same (u, v)
        from scipy.ndimage import map_coordinates
        gf = np.load(os.environ['GLOSS_FROM']); gL, gA = gf['dL_lower'], gf['dA_lower']
        GF_L, GF_A = E('GF_L', 1.0), E('GF_A', 1.0)

        def gloss_lines(x, y):
            u, v, side, inside = lip_uv(x, y)
            c_ = np.stack([np.clip(v, 0, 1) * (gL.shape[0] - 1), np.clip(u, 0, 1) * (gL.shape[1] - 1)])
            gloss_lines.da = map_coordinates(gA, c_, order=1) * GF_A * (side < 0)
            return (map_coordinates(gL, c_, order=1) * GF_L * (side < 0) * (inside > 2)).astype(np.float32)
    vis_ = visible.copy()
    dL = gloss_lines(Q[vis_, 0], Q[vis_, 1])
    dA = getattr(gloss_lines, 'da', -SPARKLE_A * dL)[dL > 0.05] if os.environ.get('GLOSS_FROM') else -SPARKLE_A * dL[dL > 0.05]; gy_, gx_ = ys[vis_][dL > 0.05], xs[vis_][dL > 0.05]; dL = dL[dL > 0.05]
    px_ = cv2.cvtColor(np.clip(out[gy_, gx_], 0, 1).astype(np.float32)[None], cv2.COLOR_RGB2LAB)[0]
    px_[:, 0] += dL; px_[:, 1] += dA
    out[gy_, gx_] = cv2.cvtColor(px_[None], cv2.COLOR_LAB2RGB)[0]
    src_ = os.path.basename(os.environ['GLOSS_FROM']) if os.environ.get('GLOSS_FROM') else '%d drawn dashes' % n_
    print('LIP_PAINT gloss lines (%s): %d texels, lightness added p99 %.1f max %.1f' % (src_, len(dL), np.percentile(dL, 99), dL.max()))
    # the check: the same lines over the painted lip as the camera sees it
    yy2, xx2 = gy[by0:by1, bx0:bx1].ravel() + 0.5, gx[by0:by1, bx0:bx1].ravel() + 0.5
    lab2 = cl.copy(); dl2 = gloss_lines(xx2, yy2); lab2[:, 0] += dl2 * 255 / 100; lab2[:, 1] += gloss_lines.da if os.environ.get('GLOSS_FROM') else -SPARKLE_A * dl2
    rgb2 = cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8).reshape(by1 - by0, bx1 - bx0, 3), cv2.COLOR_LAB2BGR).astype(np.float32)
    w2 = wl.reshape(by1 - by0, bx1 - bx0, 1)
    cv2.imwrite(f'{O}/mouthfit/gloss_lines_check.jpg', cv2.resize((crop * (1 - w2) + rgb2 * w2).astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
np.save(f'{O}/mouthfit/lip_mask.npy', Ws.astype(np.float16))   # where the lips are painted: tripo_rough.py LIP_MASK makes them glossy
Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8)).save(dst)
print('LIP_PAINT wrote', dst)
