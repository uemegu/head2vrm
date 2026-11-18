"""The nose onto the Tripo head's own nose (NOSE_FIT in run.sh).
python nose_fit.py OUT_DIR       -> OUT_DIR/nose_delta.npz (seam_apply.py format)

Why: on one test head the nose looked bulbous and the nostrils' dark showed on its surface. After tripo_fit.py the tip stood
2-3 mm short of Tripo's and its lobule 4-5 mm wider each side at the nostrils' height (seen in cross-sections),
so the tip read round; and as the skin takes each texel's colour from the closest Tripo point, the swollen alae took the dark of
Tripo's nostrils and alar creases onto their outer surface. Here every outward-facing vertex near the nose (within R mm of the
tip, MediaPipe 1 through corr.npz; above the upper lip) is pulled to the closest point of the Tripo surface where that surface faces
the same way (normals within ANGLE degrees, the point within REACH mm), its pull fading out over the last FADE mm of the region;
the nostrils' insides follow by ARAP (cotangent weights). ROUNDS rounds re-find the closest points from where the surface now is.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree

O = os.path.abspath(sys.argv[1])
E = lambda k, d: float(os.environ.get(k, d))
R, FADE, REACH, ANGLE = E('NOSE_R', 22) / 1000, E('NOSE_FADE', 7) / 1000, E('NOSE_REACH', 3.5) / 1000, E('NOSE_ANGLE', 35)
ROUNDS, ITERS, W_H = int(E('NOSE_ROUNDS', 4)), int(E('NOSE_ITERS', 6)), E('NOSE_W', 30)
CORR = os.environ.get('CORR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mpfb', os.environ.get('BASE', 'base_asian'), 'corr.npz'))

u = np.load(os.environ.get('UVMESH', f'{O}/uvmesh.npz'))
V0 = u['co'].astype(np.float64); tv = u['tv']; nV = len(V0)
d = np.load(f'{O}/tripo_in_mpfb.npz'); C, F = d['co'].astype(np.float64), d['tris']
c = np.load(CORR); cid = {int(i): k for k, i in enumerate(c['ids'])}
vt = lambda i: int(c['trip'][cid[i]][np.argmax(c['bary'][cid[i]])])
tip = V0[vt(1)]
lip_z = V0[vt(0)][2]   # the upper lip's top (MediaPipe 0): the region stays above the philtrum's lower half
print('NOSE tip vertex %d at %s; upper lip top z %.4f' % (vt(1), np.round(tip, 4), lip_z))


def vnormals(P):
    fn = np.cross(P[tv[:, 1]] - P[tv[:, 0]], P[tv[:, 2]] - P[tv[:, 0]])
    n = np.zeros_like(P)
    for k in range(3): np.add.at(n, tv[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# the Tripo surface: exact closest points (Ericson) among the triangles nearest by centroid
A_, B_, C_ = C[F[:, 0]], C[F[:, 1]], C[F[:, 2]]
FN = np.cross(B_ - A_, C_ - A_); FN /= np.maximum(np.linalg.norm(FN, axis=1, keepdims=True), 1e-12)
ctree = cKDTree((A_ + B_ + C_) / 3)


def closest(p, k=12):
    _, cand = ctree.query(p, k=k)
    best = np.full(len(p), np.inf); bp = np.zeros_like(p); bf = np.zeros(len(p), int)
    for j in range(k):
        f = cand[:, j]; a, b, cc = A_[f], B_[f], C_[f]
        ab, ac, ap = b - a, cc - a, p - a
        d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
        bp_ = p - b; d3, d4 = (ab * bp_).sum(1), (ac * bp_).sum(1)
        cp = p - cc; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
        va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
        den = np.maximum(va + vb + vc, 1e-20); v = vb / den; w = vc / den
        inside = (v >= 0) & (w >= 0) & (v + w <= 1)
        if (~inside).any():   # outside: the best of the three edges
            qs = []
            for e0, e1 in ((a, b), (b, cc), (cc, a)):
                t = np.clip(((p - e0) * (e1 - e0)).sum(1) / np.maximum(((e1 - e0) ** 2).sum(1), 1e-20), 0, 1)
                qs.append(e0 + (e1 - e0) * t[:, None])
            qs = np.stack(qs, 1); de = np.linalg.norm(qs - p[:, None], axis=2); qo = qs[np.arange(len(p)), de.argmin(1)]
            q = np.where(inside[:, None], a + ab * v[:, None] + ac * w[:, None], qo)
        else:
            q = a + ab * v[:, None] + ac * w[:, None]
        dd = np.linalg.norm(q - p, axis=1); better = dd < best
        best[better], bp[better], bf[better] = dd[better], q[better], f[better]
    return bp, best, bf


dist_tip = np.linalg.norm(V0 - tip, axis=1)
in_tri = np.zeros(nV, bool); in_tri[tv.ravel()] = True
region = in_tri & (dist_tip < R) & (V0[:, 2] > lip_z + E('NOSE_LIP', 4) / 1000)
fade = np.clip((R - dist_tip) / FADE, 0, 1) ** 2
fade *= np.clip((V0[:, 2] - lip_z - E('NOSE_LIP', 4) / 1000) / FADE, 0, 1) ** 2   # also over the region's lower edge (a hard edge left a crease above the lip)
print('NOSE region %d vertices (R %.0f mm)' % (region.sum(), R * 1000))
# the nostrils are not pulled: Tripo's are shallow painted dents, and pulling the base's nostril rims onto their floor lifted
# them 2.8 mm into the nose (a lump in the left nostril). Faces turned down (normal z under -NOSE_DOWN) beside the columella
# (|x - tip x| over NOSE_COL mm) only follow
n0 = vnormals(V0)
nostril = (n0[:, 2] < -E('NOSE_DOWN', 0.5)) & (np.abs(V0[:, 0] - tip[0]) > E('NOSE_COL', 4) / 1000)
print('NOSE nostril vertices pulled softly: %d' % (region & nostril).sum())

free = np.flatnonzero(region); fid = -np.ones(nV, int); fid[free] = np.arange(len(free))
tri_r = tv[region[tv].any(1)]
w = {}
for a_, b_, c_ in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    uu = V0[tri_r[:, a_]] - V0[tri_r[:, c_]]; vv = V0[tri_r[:, b_]] - V0[tri_r[:, c_]]
    cot = (uu * vv).sum(1) / np.maximum(np.linalg.norm(np.cross(uu, vv), axis=1), 1e-12)
    for i_, j_, cc in zip(tri_r[:, a_], tri_r[:, b_], cot):
        key = (min(i_, j_), max(i_, j_)); w[key] = w.get(key, 0) + 0.5 * cc
EK = np.array(list(w.keys())); EW = np.maximum(np.array(list(w.values())), 1e-3)
rows, cols, vals = [], [], []
for (i_, j_), w_ in zip(EK, EW):
    for a_, b_ in ((i_, j_), (j_, i_)):
        if fid[a_] >= 0:
            rows.append(fid[a_]); cols.append(fid[a_]); vals.append(w_)
            if fid[b_] >= 0: rows.append(fid[a_]); cols.append(fid[b_]); vals.append(-w_)
Lm = sp.csr_matrix((vals, (rows, cols)), shape=(len(free), len(free)))
# a little of each vertex's own place, so the free insides keep their place when nothing pulls them
Lm = Lm + sp.identity(len(free)) * 1e-9


def arap(P, wv, T):
    """ARAP from V0 with soft pulls wv (per free vertex) towards T"""
    solve = factorized((Lm + sp.diags(wv)).tocsc())
    for _ in range(ITERS):
        S = np.zeros((nV, 3, 3))
        for (i_, j_), w_ in zip(EK, EW):
            o_ = w_ * np.outer(V0[i_] - V0[j_], P[i_] - P[j_]); S[i_] += o_; S[j_] += o_
        Rm = np.tile(np.eye(3), (nV, 1, 1))
        for i_ in free:
            U, _, Vt = np.linalg.svd(S[i_]); Ri = Vt.T @ U.T
            if np.linalg.det(Ri) < 0: Vt[-1] *= -1; Ri = Vt.T @ U.T
            Rm[i_] = Ri
        rhs = np.zeros((len(free), 3))
        for (i_, j_), w_ in zip(EK, EW):
            e0 = V0[i_] - V0[j_]
            if fid[i_] >= 0:
                rhs[fid[i_]] += 0.5 * w_ * (Rm[i_] + Rm[j_]) @ e0
                if fid[j_] < 0: rhs[fid[i_]] += w_ * P[j_]
            if fid[j_] >= 0:
                rhs[fid[j_]] -= 0.5 * w_ * (Rm[i_] + Rm[j_]) @ e0
                if fid[i_] < 0: rhs[fid[j_]] += w_ * P[i_]
        rhs += wv[:, None] * T
        P[free] = np.c_[solve(rhs[:, 0]), solve(rhs[:, 1]), solve(rhs[:, 2])]
    return P


def report(P, tag):
    n = vnormals(P)
    q, dd, f = closest(P[free])
    ok = (dd < REACH) & ((n[free] * FN[f]).sum(1) > np.cos(np.radians(ANGLE)))
    sd = dd * np.sign(((P[free] - q) * n[free]).sum(1)) * 1000
    m = ok & (fade[free] > 0.5)
    print('NOSE %s: matched %d of %d, off the Tripo surface mm (inner region) p5 %.2f median %.2f p95 %.2f, |d|>1mm %d'
          % (tag, ok.sum(), len(free), np.percentile(sd[m], 5), np.median(sd[m]), np.percentile(sd[m], 95), int((np.abs(sd[m]) > 1).sum())))
    return q, ok


P = V0.copy()
for rnd in range(ROUNDS):
    q, ok = report(P, 'round %d' % rnd)
    # the nostrils: pulled softly (NOSTRIL_W) and no further than NOSTRIL_CAP mm from where they started, so the rims rise to the
    # alae's height (left free, the base's rims hung 1-2 mm under Tripo's as two stubs) but the insides do not climb into Tripo's dents
    nf = nostril[free]
    wv = W_H * fade[free] * ok * np.where(nf, E('NOSTRIL_W', 0.5), 1.0)
    T = np.where(ok[:, None], q, P[free])
    mv = T - V0[free]; ln = np.linalg.norm(mv, axis=1, keepdims=True); cap = E('NOSTRIL_CAP', 1.2) / 1000
    T = np.where(nf[:, None] & (ln > cap), V0[free] + mv / np.maximum(ln, 1e-12) * cap, T)
    P = arap(P, wv, T)
report(P, 'after')
n_ref = np.cross(V0[tv[:, 1]] - V0[tv[:, 0]], V0[tv[:, 2]] - V0[tv[:, 0]]); n_new = np.cross(P[tv[:, 1]] - P[tv[:, 0]], P[tv[:, 2]] - P[tv[:, 0]])
reg_t = region[tv].any(1)
print('folds: flipped %d, crushed %d' % (int((((n_ref * n_new).sum(1) < 0) & reg_t).sum()), int(((np.linalg.norm(n_new, axis=1) < 0.15 * np.linalg.norm(n_ref, axis=1)) & reg_t).sum())))
total = P - V0
mv_ = np.flatnonzero(np.linalg.norm(total, axis=1) > 1e-5)
np.savez(f'{O}/nose_delta.npz', delta=total, pos1=V0[mv_], off1=total[mv_], pos2=np.full((1, 3), 1e3), off2=np.zeros((1, 3)))
print('NOSE wrote %s, max move %.2f mm' % (f'{O}/nose_delta.npz', np.linalg.norm(total, axis=1).max() * 1000))
