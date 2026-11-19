"""The inward shift s(z) per side that takes the input head's front outline onto the portrait's (both MediaPipe's face oval in the
orthographic level front render / the portrait, the portrait put on the input head's eyes by a similarity). Lower face only: 0 at the eye
line, full from the cheekbone (lm 93's height) down; never outward (the portrait's longer chin is not added).
python slim_target.py HEAD_VERTS.npz LM_TRIPO.npy LM_PHOTO.npy OUT.npz [GAIN]"""
import sys, numpy as np
vf, lt, lp, out = sys.argv[1:5]; GAIN = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
V = np.load(vf)['V']; lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2; sc = (hi - lo).max() * 1.1
T = np.load(lt)[:, :2]; P = np.load(lp)[:, :2]
E = [33, 263, 133, 362]
def sim(a, b):   # least-squares similarity a -> b (2D)
    ma, mb = a.mean(0), b.mean(0); A, B = a - ma, b - mb
    za = A[:, 0] + 1j * A[:, 1]; zb = B[:, 0] + 1j * B[:, 1]; k = (np.conj(za) @ zb) / (np.conj(za) @ za)
    return lambda p: ((p[:, 0] - ma[0]) + 1j * (p[:, 1] - ma[1])) * k + (mb[0] + 1j * mb[1])
f = sim(P[E], T[E]); Pz = f(P); P2 = np.c_[Pz.real, Pz.imag]
to3 = lambda q: np.c_[c[0] + (q[:, 0] - 1000) / 2000 * sc, c[2] - (q[:, 1] - 1000) / 2000 * sc]   # x, z
L = [127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
R = [356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]
eye_z = to3(T[[33, 263]])[:, 1].mean(); cb_z = to3(T[[93, 323]])[:, 1].mean(); mid_x = to3(T[[1, 4, 6, 168]])[:, 0].mean()
zs = np.linspace(to3(T[[152]])[0, 1] - 0.02, eye_z, 60)
res = {}
for side, idx in (('L', L), ('R', R)):
    xs = {}
    for nm, Q in (('tripo', T), ('photo', P2)):
        p = to3(Q[idx]); o = np.argsort(p[:, 1]); xs[nm] = np.interp(zs, p[o, 1], np.abs(p[o, 0] - mid_x), left=np.nan)
    s = np.clip(xs['tripo'] - xs['photo'], 0, None)
    s = np.nan_to_num(s) * np.clip((eye_z - zs) / (eye_z - cb_z), 0, 1) * GAIN
    res['s_' + side] = s; res['w_' + side] = xs['tripo']
    print('SLIM', side, ' '.join('z%.3f:%.1f%%' % (z, 100 * v / w) for z, v, w in zip(zs[::6], s[::6], xs['tripo'][::6]) if np.isfinite(w)))
np.savez(out, z=zs, mid_x=mid_x, eye_z=eye_z, **res)
