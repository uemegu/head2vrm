"""Distance from each head texel (1/4 grid) to the input head's surface fitted into MPFB -> OUT/tripo_dist_q.npy (scalp_mask.py reads it)."""
import sys
import numpy as np
from scipy.spatial import cKDTree
O = sys.argv[1]
t = np.load(f'{O}/texels.npz'); pos = t['pos'][::4, ::4]; have = t['have'][::4, ::4]
tri = np.load(f'{O}/tripo_in_mpfb.npz'); tc, tt = tri['co'], tri['tris']
r = np.random.default_rng(0).dirichlet([1, 1, 1], size=(len(tt), 40))
samp = np.einsum('fsk,fkd->fsd', r, tc[tt]).reshape(-1, 3)
d = np.full(have.shape, 1.0, np.float32)
ys, xs = np.nonzero(have)   # every head texel (scalp_mask.py picks the heights)
d[ys, xs] = cKDTree(np.vstack([samp, tc])).query(pos[ys, xs])[0]
np.save(f'{O}/tripo_dist_q.npy', d)
