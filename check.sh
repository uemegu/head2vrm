#!/bin/bash
# 事前チェック：make_face.sh に要るものが揃っているか。足りないものは、何をすれば直るかを出す。
#   ./check.sh            すべて表示
#   ./check.sh --quiet    足りないものだけ表示（make_face.sh が最初に呼ぶ）
# 環境変数：BLENDER（Blender の実行ファイル）、MPFB_DATA（MPFB の素材フォルダ）で場所を変えられる。
F=$(cd "$(dirname "$0")" && pwd); M=$F/tools/mpfb
QUIET=; [ "$1" = --quiet ] && QUIET=1
B=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
MP=${MPFB_DATA:-"$HOME/Library/Application Support/Blender/5.2/extensions/.user/blender_org/mpfb/data"}
PY=$M/.venv/bin/python
NG=0
ok() { [ -n "$QUIET" ] || echo "  OK   $1"; }
ng() { echo "  NG   $1"; [ -n "$2" ] && echo "       → $2"; NG=$((NG+1)); }

[ -n "$QUIET" ] || echo "== OS とコマンド"
[ "$(uname)" = Darwin ] && ok "macOS" || ng "macOS 以外（$(uname)）" "今は macOS でだけ確かめています（md5 -q・Blender の場所）。Linux なら BLENDER と MPFB_DATA を指定し、md5 を置き換えてください"
for c in md5 python3 grep sed; do command -v $c >/dev/null && ok "$c" || ng "$c が見つからない"; done

[ -n "$QUIET" ] || echo "== Blender"
if [ -x "$B" ]; then
  V=$("$B" --version 2>/dev/null | head -1 | awk '{print $2}')
  case "$V" in 5.2*|5.3*|5.4*|5.5*|6.*) ok "Blender $V（$B）" ;; *) ng "Blender $V" "5.2 LTS で確かめています。https://www.blender.org/download/ から 5.2 を入れてください" ;; esac
  R=$("$B" -b --python-expr "
import bpy, addon_utils, importlib
try:
    import bl_ext.blender_org.mpfb.services as s; print('CHK mpfb ok')
except Exception as e: print('CHK mpfb ng', type(e).__name__)
print('CHK vrm', 'ok' if hasattr(bpy.ops.export_scene, 'vrm') and 'vrm' in dir(bpy.ops.export_scene) else 'ng')
" 2>/dev/null | grep '^CHK')
  echo "$R" | grep -q "mpfb ok" && ok "MPFB 拡張機能" || ng "MPFB 拡張機能が読めない" "Blender の 編集 > プリファレンス > 機能拡張 で「MPFB」を入れて有効にする"
  echo "$R" | grep -q "vrm ok" && ok "VRM アドオン（VRM 書き出し）" || ng "VRM アドオンが読めない" "Blender の 機能拡張 で「VRM format」を入れて有効にする"
else
  ng "Blender が見つからない（$B）" "Blender 5.2 を /Applications に入れるか、BLENDER=<実行ファイル> を指定"
fi

[ -n "$QUIET" ] || echo "== MPFB の素材（MakeHuman system assets, CC0）"
for f in skins/young_asian_female/young_lightskinned_female_diffuse3.png eyes/materials/brown_eye.png hair/ponytail01/ponytail01_diffuse.png \
         teeth/teeth_base/teeth.png tongue/tongue01/tongue01_diffuse.png eyebrows/eyebrow001/eyebrow001.png eyelashes/eyelashes01/eyelashes01.png \
         skins/young_asian_male/young_lightskinned_male_diffuse3.png; do
  [ -f "$MP/$f" ] && ok "$f" || { ng "素材がない：$MP/$f" "MPFB の「Load pack」で makehuman_system_assets（CC0）を入れる。別の場所なら MPFB_DATA=<.../mpfb/data>"; }
done
[ -n "${BASE_SKIN}" ] && { [ -f "$MP/skins/$BASE_SKIN" ] && ok "BASE_SKIN $BASE_SKIN" || ng "BASE_SKIN がない：$MP/skins/$BASE_SKIN"; }

for D in "$M"/base_*/; do
  BN=$(basename "$D"); [ -n "$QUIET" ] || echo "== 素体（tools/mpfb/$BN）"
  for f in base_arkit.dist.blend corr.npz; do [ -f "$D/$f" ] && ok "$f" || ng "$D$f がない" "配布物が欠けています。取り直してください"; done
  if [ -f "$D/base_arkit.blend" ] && [ -f "$D/.relinked" ] && [ "$(cat "$D/.relinked")" = "$MP" ]; then ok "素体の画像の付け替え済み"
  else ng "素体 $BN の画像がこのマシンの MPFB の素材を指していない" "./setup.sh を実行（tools/relink_base.py が付け替える）"; fi
done
[ -n "$QUIET" ] || echo "== Python（tools/mpfb/.venv）"
if [ -x "$PY" ]; then
  PV=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  [ "$PV" = 3.12 ] && ok "Python $PV" || ng "Python $PV" "3.12 で確かめています。./setup.sh を Python 3.12 で（PYTHON=python3.12 ./setup.sh）"
  for m in numpy scipy cv2 PIL mediapipe; do
    "$PY" -c "import $m" 2>/dev/null && ok "import $m" || ng "Python のモジュール $m がない" "./setup.sh（requirements.txt を入れる）"
  done
  "$PY" -c "from mediapipe.tasks.python import vision" 2>/dev/null && ok "mediapipe.tasks" || ng "mediapipe.tasks が読めない" "mediapipe を requirements.txt の版に（./setup.sh）"
else
  ng "仮想環境がない（$PY）" "./setup.sh を実行"
fi

[ -n "$QUIET" ] || echo "== 学習済みモデル"
T=$M/models/face_landmarker.task
[ -s "$T" ] && ok "face_landmarker.task（MediaPipe, Apache-2.0）" || ng "$T がない" "./setup.sh（Google の配布元から取る）"

if [ $NG = 0 ]; then [ -n "$QUIET" ] || echo "すべて揃っています。"; exit 0; fi
echo "足りないもの：$NG 件"; exit 1
