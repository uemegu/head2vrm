#!/bin/bash
# Input head GLB -> the MPFB base head fitted onto its surface, with the eye and mouth openings on the input's own painted lines, and
# the input's colours baked into the base's skin texture.   run.sh HEAD_GLB OUT_DIR   (called by make_face.sh; cwd: tools/tripo)
# Two stops for a person (exit 3): the lid lines (lidfit/check.jpg) and the mouth lines (mouthfit/check.jpg). Each is approved by
# writing the md5 of its check_lines.txt into `approved` next to it; changed lines ask again.
set -e -o pipefail
cd "$(dirname "$0")"
B=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
PY=../mpfb/.venv/bin/python
M=../mpfb
MP=${MPFB_DATA:-"$HOME/Library/Application Support/Blender/5.2/extensions/.user/blender_org/mpfb/data"}
BASE=${BASE:-base_asian}
GLB=$1; O=$(mkdir -p "$2" && cd "$2" && pwd)
q() { grep -E "rror|Traceback" || true; }
export KEEP_LEVEL=1 SEAM_TGT=1 SEAM_CORNERS=1 TEX_EXACT=1 SHIRT=0
uvmesh() { $B --background "$O/warped_e0.blend" --python $M/export_uv.py -- "$O/uvmesh.npz" 2>&1 | q; }
echo "== 1/7 入力の頭を読み、特徴点を取る"
$B --background --python tripo_export.py -- "$GLB" "$O" 2>&1 | grep -E "EXPORTED|rror"
$PY tripo_lm.py "$O" 2>&1 | grep -E "hits|agree|no face"
$B --background $M/$BASE/base_arkit.blend --python $M/export_uv.py -- "$O/base_uvmesh.npz" 2>&1 | q
echo "== 2/7 素体を入力の頭の表面へ合わせる"
$PY tripo_fit.py $M/$BASE/corr.npz "$O" "$O/warp.npz" $FIT_ARGS 2>&1 | grep -E "dropped|outside|iter 5|eyeball|after_mm|gap_mm|max_disp|rror|Traceback"
MOUTH_CLOSE=0.08 $B --background $M/$BASE/base_arkit.blend --python $M/apply.py -- "$O/warp.npz" "$O/warped_e0.blend" 2>&1 | q
uvmesh
echo "== 3/7 まぶたの縁と唇の合わせ目を、テクスチャに描かれた線へ"
$PY seam_fit.py "$O/view_+0.png" "$O" $M/$BASE/corr.npz 2>&1 | grep -E "^SEAM|rror|Traceback"
$B --background "$O/warped_e0.blend" --python seam_apply.py -- "$O/seam.npz" "$O/warped_e0.blend" 2>&1 | grep -E "^SEAM|rror|Traceback"
uvmesh
$B --background "$O/warped_e0.blend" --python eye_center.py -- "$O" "$O/warped_e0.blend" 2>&1 | grep -E "^EYE_CENTER|rror|Traceback"
echo "== 4/7 まぶたの線（人の確認が要る）"
CHECK=1 $PY lid_arap.py "$O" 2>&1 | grep -E "^target|^CHECK|rror|Traceback|failed"
if [ "$(md5 -q "$O/lidfit/check_lines.txt")" != "$(cat "$O/lidfit/approved" 2>/dev/null)" ]; then
  echo "LID_CHECK waiting: $O/lidfit/check.jpg と check_corners.jpg を見て、線が正しければ:"
  echo "  md5 -q $O/lidfit/check_lines.txt > $O/lidfit/approved   のあと、もう一度実行"
  exit 3
fi
$PY lid_arap.py "$O" 2>&1 | grep -E "^target|^ARAP|^round|^  cover|^folds|^wrote|rror|Traceback|failed"
$B --background "$O/warped_e0.blend" --python seam_apply.py -- "$O/lid_delta.npz" "$O/warped_e0.blend" 2>&1 | q
uvmesh
echo "== 5/7 口の線（人の確認が要る）"
CHECK=1 $PY mouth_arap.py "$O" 2>&1 | grep -E "^  mouth|^  lower|^CHECK|rror|Traceback"
if [ "$(md5 -q "$O/mouthfit/check_lines.txt")" != "$(cat "$O/mouthfit/approved" 2>/dev/null)" ]; then
  echo "MOUTH_CHECK waiting: $O/mouthfit/check.jpg を見て、線が正しければ:"
  echo "  md5 -q $O/mouthfit/check_lines.txt > $O/mouthfit/approved   のあと、もう一度実行"
  exit 3
fi
$B --background "$O/warped_e0.blend" --python mouth_keys.py -- "$O/mouth_keys.npz" 2>&1 | q
$PY mouth_arap.py "$O" 2>&1 | grep -E "^MOUTH|^  dropped|^folds|^wrote|rror|Traceback"
$B --background "$O/warped_e0.blend" --python seam_apply.py -- "$O/mouth_delta.npz" "$O/warped_e0.blend" 2>&1 | q
uvmesh
echo "== 6/7 鼻"
$PY nose_fit.py "$O" 2>&1 | grep -E "^NOSE|^folds|rror|Traceback"
$B --background "$O/warped_e0.blend" --python seam_apply.py -- "$O/nose_delta.npz" "$O/warped_e0.blend" 2>&1 | q
uvmesh
echo "== 7/7 色の焼き付けと比較画像"
$PY $M/raster.py "$O/uvmesh.npz" "$O/texels.npz" 4096 > /dev/null
$PY tripo_tex.py "$O/texels.npz" "$O/tripo_in_mpfb.npz" "$O/tripo_basecolor.png" "$O/warp.npz" "$MP/skins/${BASE_SKIN:-young_asian_female/young_lightskinned_female_diffuse3.png}" "$O/skin_tripo_4k.png"
(cd $M && IRIS_SCALE=${IRIS_SCALE:-1.0} .venv/bin/python iris.py "$O/view_+0.png" "$MP/eyes/materials/brown_eye.png" "$O/eye_ref.png" 2>&1 | grep -E "^iris" || true)
mkdir -p "$O/render_e0"
EYE="$O/eye_ref.png" $B --background "$O/warped_e0.blend" --python tripo_render.py -- "$O/tripo_in_mpfb.npz" "$O/tripo_basecolor.png" "$O/skin_tripo_4k.png" "$O/render_e0" 2>&1 | q
$PY cmp.py "$O/render_e0"
ls "$O/render_e0"/cmp_*.jpg
