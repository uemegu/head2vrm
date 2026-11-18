"""Rasterise the exported triangles in UV space: per-texel world position and normal."""
import sys
import numpy as np

def rasterize(npz, size):
    d = np.load(npz)
    co, nrm, tv, tuv = d['co'], d['nrm'], d['tv'], d['tuv']
    pos = np.zeros((size, size, 3), np.float32)
    nor = np.zeros((size, size, 3), np.float32)
    have = np.zeros((size, size), bool)
    px = tuv * size
    px[:, :, 1] = size - px[:, :, 1]  # image rows go down
    for t in range(len(tv)):
        a, b, c = px[t]
        x0, y0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        qx, qy = xs - a[0], ys - a[1]
        l1 = (qx * v1[1] - v1[0] * qy) / den
        l2 = (v0[0] * qy - qx * v0[1]) / den
        l0 = 1 - l1 - l2
        m = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not m.any():
            continue
        i, j, k = tv[t]
        w = np.stack([l0[m], l1[m], l2[m]], 1)
        yy, xx = (ys[m] - 0.5).astype(int), (xs[m] - 0.5).astype(int)
        pos[yy, xx] = w @ co[[i, j, k]]
        nor[yy, xx] = w @ nrm[[i, j, k]]
        have[yy, xx] = True
    return pos, nor, have

if __name__ == '__main__':
    pos, nor, have = rasterize(sys.argv[1], int(sys.argv[3]))
    np.savez_compressed(sys.argv[2], pos=pos, nrm=nor, have=have, eye_co=np.load(sys.argv[1])['eye_co'])
    print('texels', int(have.sum()))
