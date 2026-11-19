#!/bin/bash
# 入力の頭の GLB → MakeHuman の素体の頭をその形と色に合わせた VRM（全身の素体つき）。目・まぶた・口・舌は動く
# （ARKit の 52 キー、VRM の表情プリセット、視線）。GLB に髪の部品があれば、頭に載せて揺れるようにする（HAIR_PART）。
# HEAD_ONLY=1 で、首で切った頭だけにする（体・服・靴を外す）。
#
#   ./make_face.sh <作業フォルダ>          作業フォルダ/face.env に設定（face.env.example を参照）
#   SKIP_RUN=1 ./make_face.sh <作業フォルダ>   合わせ直しを飛ばし、色の仕上げから（線の承認や合わせはそのまま）
#
# 途中で 2 回、人の確認で止まる（exit 3）：まぶたの線と口の線。表示される画像を見て、正しければ表示されたコマンドで承認し、
# もう一度同じコマンドを実行する。
#
# 出力：作業フォルダ/<NAME>.vrm、<NAME>-closedlid.png/.json（閉じたまぶたの肌の補正。付属のビューア用）
#       作業フォルダ/out/render_e0/cmp_*.jpg（元のモデルと並べた比較）
set -e -o pipefail
F=$(cd "$(dirname "$0")" && pwd); T=$F/tools/tripo; M=$F/tools/mpfb; PL=$T/pipeline
[ -n "$1" ] || { sed -n 2,13p "$0"; exit 2; }
"$F/check.sh" --quiet || { echo "事前チェックで足りないものがあります（./check.sh で詳細）"; exit 1; }
P=$(cd "$1" && pwd); NAME=; set -a; source "$P/face.env"; set +a;   # face.env の値は、どの段（Python・Blender）にも渡る
 : "${NAME:?face.env に NAME を書いてください}"; : "${HEAD_GLB:?face.env に HEAD_GLB を書いてください}"
B=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}; PY=$M/.venv/bin/python
MP=${MPFB_DATA:-"$HOME/Library/Application Support/Blender/5.2/extensions/.user/blender_org/mpfb/data"}; export BLENDER=$B MPFB_DATA=$MP
export BASE=${BASE:-base_asian}   # 素体：base_asian（女性）か base_male（男性）
[ "$BASE" = base_male ] && export BASE_SKIN=${BASE_SKIN:-young_asian_male/young_lightskinned_male_diffuse3.png}
[ -n "$BASE_SKIN" ] && export BASE_SKIN
O=$P/out; q() { grep -E "$1" || true; }
cd "$T"

if [ -z "$SKIP_RUN" ]; then
  echo "== 準備：顔と髪の部品を見分け、顔を 1 つのメッシュに、頭の傾きを水平に"
  mkdir -p $P/raw; rm -f $P/raw/hair.glb $P/src/hair.glb
  # 顔の部品：PREP_PARTS を書いたらその最初の部品、書かなければ肌の色がいちばん多い部品（hair/parts.py）
  HEAD_PART=$( [ -n "$PREP_PARTS" ] && echo "${PREP_PARTS%%:*}" || echo auto ) HAIR_PART=$( [ -n "$HAIR_GLB" ] && echo none || echo ${HAIR_PART:-auto} ) \
    $B -b -P $T/hair/parts.py -- "$HEAD_GLB" $P/raw 2>&1 | q "^PARTS|rror|Traceback"
  if [ -n "$HAIR_GLB" ]; then   # 髪は別の GLB から（同じモデルの髪つき版。大きさと位置を顔の GLB に合わせてから載せる）
    mkdir -p $P/raw_hair
    HEAD_PART=auto HAIR_PART=${HAIR_PART:-auto} $B -b -P $T/hair/parts.py -- "$HAIR_GLB" $P/raw_hair 2>&1 | q "^PARTS|rror|Traceback"
    $B -b -P $T/hair/hair_align.py -- "$HEAD_GLB" $P/raw/parts.json "$HAIR_GLB" $P/raw_hair/parts.json $P/raw/hair.glb 2>&1 | q "^HAIR_ALIGN|rror|Traceback"
    [ -f $P/raw/hair.glb ] || { echo "HAIR_GLB の髪を顔に合わせられませんでした（上のログ）"; exit 1; }
    $PY -c "import json; a=json.load(open('$P/raw/parts.json')); b=json.load(open('$P/raw_hair/parts.json')); a['hair_rgb'] = b['hair_rgb']; json.dump(a, open('$P/raw/parts.json', 'w'), indent=1)"
  fi
  [ -n "$PREP_PARTS" ] || PREP_PARTS="$($PY -c "import json; print(json.load(open('$P/raw/parts.json'))['head'])"):0,0"
  $B -b -P $PL/prep.py -- "$HEAD_GLB" $P/raw $PREP_PARTS 2>&1 | q "^PREP|rror"
  ./level_head.sh $P/raw/head.glb $P/src $( [ -f $P/raw/hair.glb ] && echo $P/raw/hair.glb )
  HEAD=$P/src/head.glb
  if [ "${SLIM_GAIN:-0}" != 0 ]; then   # 輪郭の下半分を肖像画の輪郭へ（PORTRAIT が要る）
    : "${PORTRAIT:?SLIM_GAIN を使うときは PORTRAIT（元の肖像画）が要ります}"
    $B -b -P $PL/dump_verts.py -- $P/src/head.glb $P/src/head_verts.npz 2>&1 | q "rror"
    (cd $M && .venv/bin/python -c "
from lm import detect; import numpy as np
L=detect(['$PORTRAIT','$P/src/level/level_front.png']); np.save('$P/src/lm_photo.npy',L[0]); np.save('$P/src/lm_tripo.npy',L[1])" 2>&1 | q "rror|Trace")
    $PY $PL/slim_target.py $P/src/head_verts.npz $P/src/lm_tripo.npy $P/src/lm_photo.npy $P/src/slim_s.npz $SLIM_GAIN
    mkdir -p $P/slim; $B -b -P $PL/slim_apply.py -- $P/src/head.glb $P/src/slim_s.npz $P/slim/head.glb 2>&1 | q "^SLIM|rror"
    HEAD=$P/slim/head.glb
  fi
  mkdir -p $O/lidfit; [ -f $P/corner_adjust.txt ] && cp $P/corner_adjust.txt $O/lidfit/corner_adjust.txt
  UP_RAISE=${UP_RAISE:-0} ./run.sh $HEAD $O 2>&1 | grep -v "^/Users"
fi

echo "== 仕上げ：目元・唇・目の画像"
$PY lid_project.py "$O" "$O/skin_tripo_4k.png" "$O/skin_proj.png"
if [ "${LASHES:-painted}" = 3d ]; then
  UP_H=5.5 $PY lid_paint.py "$O" "$O/skin_proj.png" "$O/skin_painted.png"
  LN_IN=0.4 LN_OUT=0.7 $PY eyeliner.py "$O" "$O/skin_painted.png" "$O/skin_nolash.png"
else cp "$O/skin_proj.png" "$O/skin_nolash.png"; fi
if [ "${LIPS:-paint}" = paint ]; then
  ACROSS=${ACROSS:-1} GLOSS_FROM=${GLOSS_FROM-$PL/data/gloss_aoi.npz} GF_L=${GF_L:-6} $PY lip_paint.py "$O" "$O/skin_nolash.png" "$O/skin_nolash.png" | q "^LIP_PAINT|rror"
fi
$B --background "$O/warped_e0.blend" --python eye_dump_cam.py -- "$O/eyeball_cam.npz" 2>&1 | q "rror|Traceback"
$PY -c "from PIL import Image; Image.open('$O/eye_ref.png').resize((2048, 2048), Image.LANCZOS).save('$O/eye_ref_2k.png')"
KEEP=${EYE_KEEP:-1} CATCH_L=${CATCH_L:-45} $PY eye_project_cam.py "$O" "$O/eyeball_cam.npz" "$O/eye_ref_2k.png" "$O/eye_proj.png" | q "^EYE_PROJECT|rror"

echo "== 書き出し：VRM"
(cd $M
$B --background "$O/warped_e0.blend" --python $T/fixpaths.py -- "$O/vrm_base.blend" 2>&1 | q "rror|Traceback"
EYE=$O/eye_proj.png $B --background "$O/vrm_base.blend" --python settex.py -- "$O/skin_nolash.png" "$MP/hair/ponytail01/ponytail01_diffuse.png" "$O/face_only.blend" 2>&1 | q "rror|Traceback"
if [ "${LASHES:-painted}" = 3d ]; then
  LASH_TEX=1 CORE=0.8 LASH_RGB=62,40,36 W0=0.12 LO_W0=0.08 UP_N=80 LO_N=26 UP_LEN=3.9 LO_LEN=1.9 LO_ALPHA=0.6 LASH_ALPHA=0.95 \
  UP_FWD=0.6 UP_LIFT=0.6 UP_CURL=0.5 UP_BACK=0.1 UP_CLUMP=0.6 LO_CLUMP=0.5 SWEEP=1 LO_OUT=0.8 \
  $B --background "$O/face_only.blend" --python $T/lash_new.py -- "$O/face_only.blend" 2>&1 | q "^LASH_NEW|rror|Traceback"
  $B --background "$O/face_only.blend" --python lash_blink.py -- "$O/face_only.blend" 2>&1 | q "rror|Traceback"
else
  $B --background "$O/face_only.blend" --python $T/drop_lashes.py -- "$O/face_only.blend" 2>&1 | q "^DROP_LASHES|rror|Traceback"
fi
$B --background "$O/face_only.blend" --python $T/caruncle.py -- "$O" "$O/face_only.blend" 2>&1 | q "^CARUNCLE|rror|Traceback"
# 目を閉じるキーで眉の毛を動かさない（眉が低い顔で、閉じたまぶたに眉の毛が筋になって伸びた。BLINK_BROW=0 で外す）
[ "${BLINK_BROW:-1}" = 0 ] || $B --background "$O/face_only.blend" --python $T/blink_brow.py -- "$O/face_only.blend" 2>&1 | q "^BLINK_BROW|rror|Traceback"
FINAL=$O/face_only.blend
if [ -f "$P/src/hair.glb" ]; then   # 入力の髪：頭と同じ動かし方で載せ、頭の骨に付け、揺れる骨を入れ、髪の下の頭皮を根元の色に塗る
  echo "== 髪：髪の部品を載せる"
  $B --background "$FINAL" --python $T/hair/hair_graft.py -- "$P/src/hair.glb" "$O/warp.npz" "$O/hair.blend" 2>&1 | q "^HAIR|rror|Traceback"
  FINAL=$O/hair.blend
  if [ "${HAIR_SPRING:-12}" != 0 ]; then
    $B --background "$FINAL" --python $T/hair/hair_spring.py -- "$FINAL" "${HAIR_SPRING:-12}" 2>&1 | q "^SPRING|rror|Traceback"
  fi
  .venv/bin/python -I $T/hair/tripo_dist.py "$O"
  $B --background "$FINAL" --python $T/hair/scalp_mask.py -- "$O" "$O/scalp_mask.npy" 2>&1 | q "^SCALP|rror|Traceback"
  ROOT_RGB=${ROOT_RGB:-$(.venv/bin/python -c "import json; v=json.load(open('$P/raw/parts.json'))['hair_rgb']; print(','.join('%g' % min(c + 12, 255) for c in v))")} \
    .venv/bin/python -I $T/hair/scalp_paint.py "$O" "$O/skin_nolash.png" "$O/skin_nolash.png" | q "^SCALP|rror|Traceback"
fi
LIP_MASK=$( [ -f "$O/mouthfit/lip_mask.npy" ] && [ "${LIPS:-paint}" = paint ] && echo "$O/mouthfit/lip_mask.npy" ) \
  .venv/bin/python $T/tripo_rough.py "$O/texels.npz" "$O/skin_nolash.png" "$O/warp.npz" "$O/roughness_4k.png"
if [ -n "$HEAD_ONLY" ]; then   # 首で切った頭だけ（体・服・靴を外す）
  $B --background "$FINAL" --python $F/tools/head_only.py -- "$O/head_only.blend" 2>&1 | q "^HEAD_ONLY|rror|Traceback"
  FINAL=$O/head_only.blend
fi
NO_HAIR=1 LOOKAT_EXPR=1 VRM_NAME=${VRM_NAME:-$NAME} VRM_AUTHOR=${VRM_AUTHOR:-unknown} ROUGHNESS=$O/roughness_4k.png \
  $B --background "$FINAL" --python export_vrm.py -- "$P/$NAME.vrm" 2>&1 | q "^EXPORT|^LOOKAT|rror|Traceback"
# 閉じたまぶたの肌の補正（描かれたまつ毛の先を消した肌。付属のビューアが目の閉じ具合に合わせて混ぜる。VRM はこれなしで完結）
$B --background "$O/face_only.blend" --python $PL/closed_lid_vis.py -- "$O/closedlid" 2>&1 | q "rror|Traceback"
.venv/bin/python $PL/closed_lid_patch.py "$O" "$O/skin_nolash.png" "$P" $NAME-closedlid 2>&1 | q "^CLOSED_PATCH|rror"
)
ls -la "$P/$NAME.vrm"
echo "できました：$P/$NAME.vrm（確認は viewer/index.html。比較画像は $O/render_e0/cmp_*.jpg）"
