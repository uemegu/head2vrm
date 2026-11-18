#!/bin/bash
# 一度だけの準備：Python の仮想環境、MediaPipe の顔の特徴点モデル、素体の画像の付け替え。最後に check.sh を通す。
#   ./setup.sh                 PYTHON=python3.12 ./setup.sh （Python の場所を変えるとき）
# Blender 5.2・MPFB 拡張機能（と MakeHuman system assets）・VRM アドオンは自分で入れる（check.sh が案内する）。
set -e
F=$(cd "$(dirname "$0")" && pwd); M=$F/tools/mpfb
B=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
MP=${MPFB_DATA:-"$HOME/Library/Application Support/Blender/5.2/extensions/.user/blender_org/mpfb/data"}
PYTHON=${PYTHON:-python3.12}
if [ ! -x "$M/.venv/bin/python" ]; then
  command -v "$PYTHON" >/dev/null || { echo "$PYTHON が見つかりません（Python 3.12 を入れるか PYTHON=... を指定）"; exit 1; }
  echo "== 仮想環境を作る（$M/.venv）"; "$PYTHON" -m venv "$M/.venv"
fi
echo "== Python のパッケージ"; "$M/.venv/bin/python" -m pip install -q -r "$F/requirements.txt"
mkdir -p "$M/models"
if [ ! -s "$M/models/face_landmarker.task" ]; then
  echo "== MediaPipe の Face Landmarker モデル（約 3.6MB、Apache-2.0）"
  curl -fL --retry 3 -o "$M/models/face_landmarker.task" \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
fi
if [ -x "$B" ] && [ -d "$MP" ]; then
  for D in "$M"/base_*/; do   # base_asian（女性）・base_male（男性）
    echo "== 素体 $(basename "$D") の画像をこのマシンの MPFB の素材へ付け替える（base_arkit.dist.blend → base_arkit.blend）"
    cp "$D/base_arkit.dist.blend" "$D/base_arkit.blend"
    "$B" -b "$D/base_arkit.blend" --python "$F/tools/relink_base.py" -- "$MP" 2>&1 | grep -E "^RELINK (MISSING|done)" || true
    rm -f "$D/base_arkit.blend1"; printf '%s' "$MP" > "$D/.relinked"
  done
fi
"$F/check.sh"
