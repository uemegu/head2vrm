"""Fit one smooth Gaussian-RBF warp so the basemesh landmark points land on the triangulated orbit landmarks.

python fit.py CORR_NPZ POINTS3D_NPY OUT_NPZ [--sigma 0.014 --centers 300 --mu 1e-4]
The warp is linear in its weights, so this is a single regularised least-squares solve.
"""
import argparse
import json

import numpy as np

OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE_TIP, CHIN = 1, 152


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d0.T @ s0 / len(src))
    fix = np.eye(3)
    fix[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ fix @ vt
    scale = np.trace(np.diag(sig) @ fix) / ((s0 ** 2).sum(1).mean())
    return scale, rot, mu_d - scale * rot @ mu_s


def farthest(points, count):
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(0), axis=1)))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(count - 1):
        chosen.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return points[chosen]


class Silhouettes:
    """Real face edges (MediaPipe face-skin segmentation) of a few orbit frames, from below the ears to the chin,
    and the matching mesh silhouette: vertices whose normal is side-on to that frame's camera."""

    def __init__(self, run, frames, s, R, t, under_chin=0.0):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from pathlib import Path
        from scipy.spatial import cKDTree
        self.cam = json.load(open(run + '/camera.json'))
        marks = np.load(run + '/marks2d.npy')
        cap = cv2.VideoCapture(run + '/orbit.mp4')
        imgs = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            imgs.append(f)
        model = str(Path(__file__).parent / 'models' / 'selfie_multiclass.tflite')
        opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model), output_category_mask=True)
        self.s, self.R, self.t = s, R, t
        self.frames = []
        with vision.ImageSegmenter.create_from_options(opts) as seg:
            for f in frames:
                if str(f) not in self.cam['per_frame'] or np.isnan(marks[f, 0, 0]):
                    continue
                rgb = cv2.cvtColor(imgs[f], cv2.COLOR_BGR2RGB)
                cat = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))).category_mask.numpy_view()
                face = (cat == 3).astype(np.uint8)
                n, lab, stats, _ = cv2.connectedComponentsWithStats(face)
                face = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
                cnt = max(cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0, :].astype(float)
                nose_v, chin_v = marks[f, 2, 1], marks[f, 152, 1]
                v_lo = nose_v - 0.15 * (chin_v - nose_v)          # from about the ear lobes (the jaw angle) down
                # the ears hang outside the cheek line: keep the edge between the face-oval points in front of them
                x_lo, x_hi = sorted([marks[f, 234, 0], marks[f, 454, 0]])
                ang = self.cam['per_frame'][str(f)]['angle_deg']
                v_hi = chin_v - 2
                keep = (cnt[:, 1] > v_lo) & (cnt[:, 0] > x_lo - 4) & (cnt[:, 0] < x_hi + 4)
                if under_chin > 0 and abs(ang) > 30:
                    # turned frames show the line under the chin (face skin against the neck): keep it, minus the ear side
                    v_hi = chin_v + under_chin * (chin_v - nose_v)
                    toward = 1 if ang < 0 else -1          # the chin points to image left when the angle is negative
                    chin_u = marks[f, 152, 0]
                    under = (cnt[:, 1] > chin_v - 2) & (cnt[:, 1] < v_hi) & (toward * (chin_u - cnt[:, 0]) > -0.6 * (x_hi - x_lo))
                    keep |= under
                edge = cnt[keep & (cnt[:, 1] < v_hi + 2)]
                self.frames.append((f, cKDTree(edge), v_lo, v_hi))

    def project(self, P, pf):
        c = self.cam
        Qb = ((P - self.t) @ self.R) / self.s
        Q = np.stack([Qb[:, 0], Qb[:, 2], -Qb[:, 1]], 1)
        ang, e = np.radians(pf['angle_deg']), c['elevation']
        ca, sa, ce, se = np.cos(ang), np.sin(ang), np.cos(e), np.sin(e)
        r = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]]) @ np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        p = Q @ r.T
        z = c['dist'] - p[:, 2]
        f = pf['zoom'] * c['focal']
        u = c['size'] / 2 + pf['du'] + f * (p[:, 0] + c['tx']) / z
        v = c['size'] / 2 + pf['dv'] - f * (p[:, 1] + c['ty']) / z
        cam_q = r.T @ np.array([-c['tx'], -c['ty'], c['dist']])
        cam_b = self.s * (self.R @ np.array([cam_q[0], -cam_q[2], cam_q[1]])) + self.t
        return u, v, z, r, f, cam_b

    def unproject(self, u, v, z, r, pf):
        c = self.cam
        f = pf['zoom'] * c['focal']
        px = (u - c['size'] / 2 - pf['du']) * z / f - c['tx']
        py = -(v - c['size'] / 2 - pf['dv']) * z / f - c['ty']
        p = np.stack([px, py, c['dist'] - z], 1)
        Q = p @ r                                         # r is orthonormal: inverse = transpose
        return (self.s * (self.R @ np.stack([Q[:, 0], -Q[:, 2], Q[:, 1]], 1).T)).T + self.t

    def targets(self, X, tris, cand):
        nrm = np.zeros_like(X)
        fn = np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]])
        for k in range(3):
            np.add.at(nrm, tris[:, k], fn)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        idx_all, tg_all = [], []
        for f, tree, v_lo, v_hi in self.frames:
            pf = self.cam['per_frame'][str(f)]
            u, v, z, r, foc, cam_b = self.project(X[cand], pf)
            view = cam_b - X[cand]
            view /= np.linalg.norm(view, axis=1, keepdims=True)
            side = np.abs((nrm[cand] * view).sum(1)) < 0.12
            ok = side & (v > v_lo) & (v < v_hi)
            if not ok.any():
                continue
            d, j = tree.query(np.stack([u[ok], v[ok]], 1), distance_upper_bound=30)
            good = np.isfinite(d)
            uv = tree.data[j[good]]
            sel = np.where(ok)[0][good]
            tg = self.unproject(uv[:, 0], uv[:, 1], z[sel], r, pf)
            idx_all.append(cand[sel])
            tg_all.append(tg)
        if not idx_all:
            return np.zeros(0, int), np.zeros((0, 3))
        return np.concatenate(idx_all), np.concatenate(tg_all)


class Openings(Silhouettes):
    """Eye outlines (MediaPipe's 16-point lid contours) of a few near-front frames; each rim vertex goes to the
    nearest point of the densified outline, at its own depth."""
    EYES = ([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466])

    def __init__(self, run, frames, s, R, t):
        from scipy.spatial import cKDTree
        self.cam = json.load(open(run + '/camera.json'))
        marks = np.load(run + '/marks2d.npy')
        self.s, self.R, self.t = s, R, t
        self.frames = []
        for f in frames:
            if str(f) not in self.cam['per_frame'] or np.isnan(marks[f, 0, 0]):
                continue
            pts = []
            for ids in self.EYES:
                loop = marks[f, ids + ids[:1], :2]
                for a_, b_ in zip(loop[:-1], loop[1:]):
                    pts.append(a_ + (b_ - a_) * np.linspace(0, 1, 8, endpoint=False)[:, None])
            self.frames.append((f, cKDTree(np.concatenate(pts))))

    def targets(self, X, rim):
        idx_all, tg_all = [], []
        for f, tree in self.frames:
            pf = self.cam['per_frame'][str(f)]
            u, v, z, r, foc, cam_b = self.project(X[rim], pf)
            d, j = tree.query(np.stack([u, v], 1), distance_upper_bound=12)
            good = np.isfinite(d)
            uv = tree.data[j[good]]
            tg_all.append(self.unproject(uv[:, 0], uv[:, 1], z[good], r, pf))
            idx_all.append(rim[good])
        return np.concatenate(idx_all), np.concatenate(tg_all)


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def phi(x, centers, sigma):
    d2 = ((x[:, None, :] - centers[None]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sigma ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corr')
    ap.add_argument('points')
    ap.add_argument('out')
    ap.add_argument('--sigma', type=float, default=0.014)
    ap.add_argument('--centers', type=int, default=300)
    ap.add_argument('--mu', type=float, default=1e-4)
    ap.add_argument('--anchor', type=float, default=3.0)
    ap.add_argument('--strength', type=float, default=1.0)
    ap.add_argument('--lap', type=float, default=30.0)
    ap.add_argument('--rest', type=float, default=0.01)
    ap.add_argument('--contour', type=float, default=0.5)
    ap.add_argument('--run', default='')
    ap.add_argument('--sil', type=float, default=0.0, help='weight of the jaw silhouette term (needs --run)')
    ap.add_argument('--sil-frames', default='0,3,6,66,69')
    ap.add_argument('--sil-iters', type=int, default=4)
    ap.add_argument('--under-chin', type=float, default=0.0, help='also match the edge under the chin (side frames), as a fraction of nose-to-chin')
    ap.add_argument('--extra', default='', help='npz with rest/target point pairs (image-space corrections) and weight')
    ap.add_argument('--eye-weight-fit', type=float, default=1.0, help='extra weight of the 32 eye-outline landmarks (lid lines, corners)')
    ap.add_argument('--symmetric', action='store_true', help='mirror every constraint across x = 0 (the base mesh is symmetric)')
    ap.add_argument('--openings', type=float, default=0.0, help='weight of the eye-hole rim -> image eye outline term')
    a = ap.parse_args()

    c = np.load(a.corr)
    co, trip, bary, ids = c['co'], c['trip'], c['bary'], c['ids']
    src_all = (co[trip] * bary[:, :, None]).sum(1)
    ref = np.load(a.points)
    ref_b = np.stack([ref[:, 0], -ref[:, 2], ref[:, 1]], 1)  # orbit (x right, y up, z to camera) -> Blender

    keep = [k for k, i in enumerate(ids) if i < 468 and (i not in OVAL or i == CHIN) and not np.isnan(ref_b[i, 0])]
    lm = ids[keep]
    src = src_all[keep]
    s, R, t = umeyama(ref_b[lm], src)
    tgt = (s * (R @ ref_b[lm].T)).T + t
    before = np.linalg.norm(tgt - src, axis=1)

    nose = src[list(lm).index(NOSE_TIP)]
    chin = src[list(lm).index(CHIN)]
    in_body = c['in_body']
    body = np.where(in_body)[0]
    head = body[(co[body, 2] > chin[2] - 0.07)]
    hc = co[head].mean(0)
    face = head[(np.linalg.norm(co[head] - (nose + np.array([0, 0.03, 0.01])), axis=1) < 0.105) & (co[head, 1] < hc[1] + 0.01)]
    jaw = head[(co[head, 2] > chin[2] - (0.04 if a.under_chin > 0 else 0.01)) & (co[head, 2] < nose[2]) & (co[head, 1] < hc[1] + 0.07)]  # up to the jaw angle under the ears
    if a.sil > 0:
        face = np.union1d(face, jaw)
    centers = farthest(co[face], a.centers)
    anchors = head[(co[head, 1] > hc[1] + 0.04) | (co[head, 2] > nose[2] + 0.11)]
    anchors = anchors[::3]
    z_lo, z_hi = (chin[2] - 0.09, chin[2] - 0.045) if a.under_chin > 0 else (chin[2] - 0.06, chin[2] - 0.015)

    # Face oval from the front frame: the 2D outline fixes x/height, depth stays the basemesh's own.
    oval_src, oval_tgt = np.zeros((0, 3)), np.zeros((0, 3))
    if a.run and a.contour > 0:
        cam = json.load(open(a.run + '/camera.json'))
        m0 = np.load(a.run + '/marks2d.npy')[0]
        pf = cam['per_frame']['0']
        oi = [k for k, i in enumerate(ids) if i in OVAL and i != CHIN and not np.isnan(m0[i, 0])]
        oval_src = src_all[oi]
        qb = ((oval_src - t) @ R) / s
        zq = -qb[:, 1]
        D, f = cam['dist'], cam['focal'] * pf['zoom']
        u, v = m0[ids[oi], 0] - cam['size'] / 2 - pf['du'], m0[ids[oi], 1] - cam['size'] / 2 - pf['dv']
        xq = u * (D - zq) / f - cam['tx']
        yq = -v * (D - zq) / f - cam['ty']
        oval_tgt = (s * (R @ np.stack([xq, -zq, yq], 1).T)).T + t
        print('oval before mm', float(np.median(np.linalg.norm(oval_tgt - oval_src, axis=1)) * 1000))

    def basis(x):
        return smoothstep(x[:, 2], z_lo, z_hi)[:, None] * phi(x, centers, a.sigma)

    # Uniform Laplacian of the displacement over the head: the warp may move the face, but smoothly.
    pos = {v: k for k, v in enumerate(head)}
    nbr = [set() for _ in head]
    for tri in c['tris']:
        if all(v in pos for v in tri):
            for x in tri:
                nbr[pos[x]].update(pos[y] for y in tri if y != x)
    Bh = basis(co[head])
    L = Bh - np.stack([Bh[list(n)].mean(0) if n else Bh[k] for k, n in enumerate(nbr)])
    MX = np.array([-1.0, 1.0, 1.0])

    def mirrored(sr, tg):
        return (np.vstack([sr, sr * MX]), np.vstack([tg, tg * MX])) if a.symmetric else (sr, tg)

    EYE_IDS = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
               263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466}
    lw = np.sqrt(np.where(np.isin(lm, list(EYE_IDS)), a.eye_weight_fit, 1.0))[:, None]
    if a.symmetric:
        lw = np.vstack([lw, lw])

    def solve(extra_src, extra_tgt, w_extra):
        l_src, l_tgt = mirrored(src, tgt)
        o_src, o_tgt = mirrored(oval_src, oval_tgt)
        extra_src, extra_tgt = mirrored(extra_src, extra_tgt)
        A = np.vstack([lw * basis(l_src), np.sqrt(a.contour) * basis(o_src), np.sqrt(w_extra) * basis(extra_src),
                       np.sqrt(a.anchor) * basis(co[anchors]), np.sqrt(a.mu) * np.eye(len(centers)),
                       np.sqrt(a.lap) * L, np.sqrt(a.rest) * Bh])
        B = np.vstack([lw * (l_tgt - l_src), np.sqrt(a.contour) * (o_tgt - o_src), np.sqrt(w_extra) * (extra_tgt - extra_src),
                       np.zeros((len(anchors), 3)), np.zeros((len(centers), 3)),
                       np.zeros((len(head), 3)), np.zeros((len(head), 3))])
        return np.linalg.lstsq(A, B, rcond=None)[0]

    fixed_src, fixed_tgt, fixed_w = np.zeros((0, 3)), np.zeros((0, 3)), 0.0
    if a.extra:
        ex = np.load(a.extra)
        fixed_src, fixed_tgt, fixed_w = ex['rest'], ex['target'], float(ex['weight'])
        print('image-space corrections', len(fixed_src))
    plain_solve = solve

    def solve(extra_src, extra_tgt, w_extra):
        """plain_solve plus the fixed image-space corrections, each group keeping its own weight."""
        if not len(fixed_src):
            return plain_solve(extra_src, extra_tgt, w_extra)
        # scale both groups' rows into one generic term of weight 1
        r_src = np.vstack([extra_src, fixed_src])
        r_w = np.concatenate([np.full(len(extra_src), np.sqrt(w_extra)), np.full(len(fixed_src), np.sqrt(fixed_w))])[:, None]
        l_src, l_tgt = mirrored(src, tgt)
        o_src, o_tgt = mirrored(oval_src, oval_tgt)
        e_src, e_tgt = mirrored(r_src, np.vstack([extra_tgt, fixed_tgt]))
        e_w = np.vstack([r_w, r_w]) if a.symmetric else r_w
        A = np.vstack([lw * basis(l_src), np.sqrt(a.contour) * basis(o_src), e_w * basis(e_src),
                       np.sqrt(a.anchor) * basis(co[anchors]), np.sqrt(a.mu) * np.eye(len(centers)),
                       np.sqrt(a.lap) * L, np.sqrt(a.rest) * Bh])
        B = np.vstack([lw * (l_tgt - l_src), np.sqrt(a.contour) * (o_tgt - o_src), e_w * (e_tgt - e_src),
                       np.zeros((len(anchors), 3)), np.zeros((len(centers), 3)),
                       np.zeros((len(head), 3)), np.zeros((len(head), 3))])
        return np.linalg.lstsq(A, B, rcond=None)[0]

    W = solve(np.zeros((0, 3)), np.zeros((0, 3)), 0.0)
    if a.openings > 0 and a.run:
        # lid margins: skin near the eyes that turns inward (towards the eyeball) as seen from the front,
        # but not the socket behind it (MakeHuman's lids fold into a closed socket, there is no hole edge)
        tr = c['tris']
        nrm = np.zeros_like(co)
        fn = np.cross(co[tr[:, 1]] - co[tr[:, 0]], co[tr[:, 2]] - co[tr[:, 0]])
        for k3 in range(3):
            np.add.at(nrm, tr[:, k3], fn)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        rim = []
        for ids in ((33, 133, 159, 145), (362, 263, 386, 374)):
            pts = src[[list(lm).index(i) for i in ids]]
            ctr = pts.mean(0)
            half_w = np.linalg.norm(pts[0] - pts[1]) / 2
            d = co[body] - ctr
            near = (np.abs(d[:, 0]) < half_w * 1.1) & (np.abs(d[:, 2]) < half_w * 0.75)
            front = d[:, 1] < 0.003                         # Blender -Y is the front: not deeper than the corners
            inward = nrm[body, 1] > -0.35                   # normal no longer facing the viewer
            rim.append(body[near & front & inward])
        rim = np.concatenate(rim)
        np.save(a.out.replace('.npz', '_rim.npy'), rim)
        print('lid-margin vertices', len(rim))
        op = Openings(a.run, [0, 3, 69], s, R, t)
        sil_extra = []
        for it in range(3):
            cur = co + basis(co) @ W
            idx, tg = op.targets(cur, rim)
            print('openings iter', it, 'points', len(idx), 'median gap mm', round(float(np.median(np.linalg.norm(tg - cur[idx], axis=1)) * 1000), 2))
            W = solve(co[idx], co[idx] + (tg - cur[idx]), a.openings * len(src) / max(len(idx), 1))
        open_src, open_tgt = co[idx], co[idx] + (tg - cur[idx])
    if a.sil > 0 and a.run:
        sil = Silhouettes(a.run, [int(f) for f in a.sil_frames.split(',')], s, R, t, a.under_chin)
        lower = jaw
        for it in range(a.sil_iters):
            cur = co + basis(co) @ W
            idx, tg = sil.targets(cur, c['tris'], lower)
            # rest positions + the displacement those points need
            ex_src, ex_tgt = co[idx], co[idx] + (tg - cur[idx])
            w = a.sil * len(src) / max(len(idx), 1)
            print('silhouette iter', it, 'points', len(idx), 'median gap mm', round(float(np.median(np.linalg.norm(tg - cur[idx], axis=1)) * 1000), 2))
            W = solve(ex_src, ex_tgt, w)
    W *= a.strength
    after = np.linalg.norm(src + basis(src) @ W - tgt, axis=1)
    eye_k = np.isin(lm, [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466])
    print('eye outline residual mm median', round(float(np.median(after[eye_k]) * 1000), 2), 'max', round(float(after[eye_k].max() * 1000), 2))
    disp = np.linalg.norm(basis(co[head]) @ W, axis=1)
    # eye openings grow/shrink with the warp: scale each eyeball by its corner-to-corner width change
    eye_scale = []
    for ca, cb in ((33, 133), (362, 263)):
        ka, kb = list(lm).index(ca), list(lm).index(cb)
        wa = src + basis(src) @ W
        eye_scale.append(float(np.linalg.norm(wa[ka] - wa[kb]) / np.linalg.norm(src[ka] - src[kb])))
    print('eye scale', eye_scale)
    if len(oval_src):
        print('oval after mm', float(np.median(np.linalg.norm(oval_src + basis(oval_src) @ W - oval_tgt, axis=1)) * 1000))
    np.savez(a.out, eye_scale=np.array(eye_scale), centers=centers, W=W, sigma=a.sigma, z_lo=z_lo, z_hi=z_hi, scale=s, rot=R, trans=t)
    print(json.dumps({'landmarks': len(lm), 'scale': s, 'face_verts': len(face), 'anchors': len(anchors),
                      'before_mm_median': float(np.median(before) * 1000), 'before_mm_p90': float(np.percentile(before, 90) * 1000),
                      'after_mm_median': float(np.median(after) * 1000), 'after_mm_p90': float(np.percentile(after, 90) * 1000),
                      'max_disp_mm': float(disp.max() * 1000)}, indent=1))


if __name__ == '__main__':
    main()
