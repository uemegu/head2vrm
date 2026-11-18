"""MediaPipe face landmarks (478) of images -> NPY (pixels, z in pixel units). Optional overlay JPG."""
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL = str(Path(__file__).parent / 'models' / 'face_landmarker.task')


def detect(paths):
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL), num_faces=1)
    out = []
    with vision.FaceLandmarker.create_from_options(opts) as det:
        for p in paths:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
            if not res.face_landmarks:
                out.append(np.full((478, 3), np.nan))
                continue
            out.append(np.array([[l.x * w, l.y * h, l.z * w] for l in res.face_landmarks[0]]))
    return np.stack(out)


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    marks = detect([src])[0]
    np.save(dst, marks)
    img = cv2.imread(src)
    for x, y, _ in marks:
        if not np.isnan(x):
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite(dst.replace('.npy', '.jpg'), img)
    print('found', int((~np.isnan(marks[:, 0])).sum()))
