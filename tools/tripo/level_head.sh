#!/bin/bash
# Any Tripo head GLB -> OUT_DIR/head.glb turned level, the angle measured, not given.   level_head.sh GLB OUT_DIR [HAIR_GLB]   (cwd: tripo/)
# HAIR_GLB (hair/parts.py, in the head part's frame): turned by the same angle about the same point -> OUT_DIR/hair.glb
# 1. an orthographic front render (level_front.py), MediaPipe's landmarks on it, each lifted onto the mesh by a ray (level_lm3d.py)
# 2. the lean: the mean angle in the face's plane of the left-right pairs (eye corners outer and inner, brows, mouth corners, cheeks)
# 3. the whole head turned by it about the point between the eyes (level_prep.py), then measured again (must be under 1 degree)
set -e -o pipefail
B=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}; P=$PWD/../mpfb/.venv/bin/python
GLB=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); O=$(mkdir -p "$2" && cd "$2" && pwd); C=$O/level; mkdir -p $C
measure() {   # GLB TAG -> prints "angle cx cz"
  $B -b -P level_front.py -- "$1" $C/$2_front.png 2>&1 | grep -E "rror" || true
  (cd ../mpfb && .venv/bin/python -c "
from lm import detect; import numpy as np; np.save('$C/$2_lm.npy', detect(['$C/$2_front.png'])[0])" 2>&1 | grep -E "rror|Trace" || true)
  $B -b -P level_lm3d.py -- "$1" $C/$2_lm.npy $C/$2_lm3d.npy 2>&1 | grep -E "rror" || true
  $P -c "
import numpy as np; P=np.load('$C/$2_lm3d.npy')
a=[np.degrees(np.arctan2(*(P[b]-P[a])[[2,0]])) for a,b in ((33,263),(133,362),(70,300),(61,291),(234,454))]
a=[v for v in a if np.isfinite(v)]; c=P[[33,263,133,362]].mean(0)
print('%.3f %.5f %.5f' % (np.mean(a), c[0], c[2]))"
}
read A CX CZ <<< "$(measure "$GLB" raw)"
echo "LEVEL lean measured $A deg"
LEVEL=$(python3 -c "print(-($A))") LEVEL_C=$CX,$CZ $B -b --python level_prep.py -- "$GLB" "$O" 2>&1 | grep -E "^LEVEL|rror"
read A2 _ _ <<< "$(measure "$O/head.glb" level)"
echo "LEVEL after: $A2 deg"
echo "$A $CX $CZ" > $C/level.txt
if [ -n "$3" ]; then
  mkdir -p $O/hair_level
  LEVEL=$(python3 -c "print(-($A))") LEVEL_C=$CX,$CZ $B -b --python level_prep.py -- "$3" "$O/hair_level" 2>&1 | grep -E "^LEVEL|rror"
  mv "$O/hair_level/head.glb" "$O/hair.glb"; rmdir "$O/hair_level"; echo "LEVEL hair turned the same: $O/hair.glb"
fi
python3 -c "import sys; sys.exit(0 if abs($A2) < 1 else 1)" || { echo "LEVEL failed: still $A2 deg"; exit 1; }
