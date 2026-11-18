# face-maker：Tripo の頭から、表情が動く VRM を作る

[Tripo](https://www.tripo3d.ai/) で作った頭の 3D モデル（GLB）から、目・口・表情が動く VRM 1.0 のアバターを作ります。

作り方の考え方は、形と色を Tripo から、動きを素体から、というものです。MakeHuman の素体（MPFB で組んだ、ARKit の表情キー入りの体）の頭を、Tripo の頭の形に合わせて変形し、Tripo の肌と目の色を焼き付けます。顔の見た目は Tripo のまま、まばたき・口の形・表情は素体の仕組みで動きます。

## できること

- **目**：まばたき（左右別々）と視線。視線は VRM の表情キー方式なので、どのアプリでも見る先に合わせて目が動きます。黒目と白目は Tripo の絵を眼球へ写し、角膜にはつやのある透明な層を付けています。
- **口**：あいうえお、舌、あごの開閉。
- **表情**：VRM の標準の表情（笑顔・怒り・悲しみ・驚き・穏やか）と、ARKit の 52 個。iPhone の顔のトラッキングの値でそのまま動かせます。
- **髪**：GLB に髪が別の部品として入っていれば、頭に載せます。揺れるように骨と VRM の揺れもの（SpringBone）を付け、髪の下の頭皮は髪の根元の色に塗ります。
- **体**：素体の体・服・靴が付いた全身の VRM になります。`HEAD_ONLY=1` にすると、首で切った頭だけになります。

1 人あたり数分です（M シリーズの Mac）。途中で 2 回、まぶたの線と口の線を人が確認するために止まります。

## 必要なもの

| もの | 確かめた版 | 入れ方 |
|---|---|---|
| macOS | 26 | — |
| Blender | 5.2 LTS | https://www.blender.org/download/ から `/Applications` へ。別の場所なら `BLENDER=<実行ファイル>` |
| MPFB（Blender の拡張機能） | 2.x | Blender の「編集 > プリファレンス > 機能拡張」で MPFB を入れて有効にし、MPFB の「Load pack」で **MakeHuman system assets**（CC0）を入れる |
| VRM format（Blender のアドオン） | 3.x | 同じく「機能拡張」で VRM format を入れて有効にする |
| Python | 3.12 | `setup.sh` が `tools/mpfb/.venv` に仮想環境を作り、`requirements.txt` を入れる |
| MediaPipe Face Landmarker のモデル | float16/1 | `setup.sh` が Google の配布元から取る（約 3.6MB） |

## 使い方

```bash
./setup.sh          # 最初に一度だけ。最後に ./check.sh（事前チェック）が走る
./check.sh          # 足りないものと、その直し方を表示する
```

1. 作業フォルダを作り（例：`work/myface/`）、`face.env.example` を `face.env` という名前でそこへコピーして、`NAME` と `HEAD_GLB` を書きます。`work/` の下はリポジトリの管理から外れます。
2. 実行します。

   ```bash
   ./make_face.sh work/myface
   ```

3. **まぶたの線の確認で止まります（終了コード 3）。** `work/myface/out/lidfit/check.jpg` と `check_corners.jpg` を見て、線が目の縁に乗っていれば、表示されたコマンド（`md5 -q … > …/approved`）で承認して、同じコマンドをもう一度実行します。目頭・目尻が少しずれているときは `corner_adjust.txt` で、上まぶたの線全体が低いときは `UP_RAISE` で直せます（下の「設定」）。
4. **口の線の確認で止まります。** `out/mouthfit/check.jpg` を見て、同じように承認して再実行します。
5. できあがりは `work/myface/myface.vrm` です。`out/render_e0/cmp_*.jpg` に、Tripo（上）と作ったもの（下）を並べた比較画像が出ます。

承認は線の中身に対して記録されます。入力や設定を変えて線が変わると、もう一度確認を求められます。合わせ直しを飛ばして、色の仕上げからやり直すときは `SKIP_RUN=1 ./make_face.sh work/myface` です。

### 確認用のビューア

```bash
python3 -m http.server 5203 --bind 127.0.0.1
```

`http://localhost:5203/viewer/?model=../work/myface/myface.vrm` を開きます（VRM ファイルを画面へドロップしても開けます）。表情のスライダー、ARKit の 52 個、自動まばたきが使えます。目は見ている人（カメラ）の方を向きます。

## 入力の Tripo の頭

- 正面を向き、口を閉じ、目を開けた顔にしてください。目は頭に描き込まれていてかまいません。まぶたの開きは Tripo が描いた線に合わせ、黒目と白目はその絵を眼球へ写します。
- 髪は、別の部品になっていれば移植されます（部品の色から自動で見分けます。外れるときは `HAIR_PART`）。髪が頭と一体の 1 部品だと、髪の色が頭の形と肌に焼き付くので、坊主頭か髪を別部品にしたものを使ってください。
- 顔の出来は、ほぼ Tripo の時点で決まります。作り始める前に Tripo の顔をよく見て、納得できるものを使ってください。後の工程で顔を別人のように変えることはしません。

## 設定（face.env）

作業フォルダの `face.env` に書きます。書かなかったものは既定値になります。

| 名前 | 既定 | 意味 |
|---|---|---|
| `NAME` | （必須） | 出力の名前（`<作業フォルダ>/<NAME>.vrm`） |
| `HEAD_GLB` | （必須） | Tripo の頭の GLB |
| `BASE` | `base_asian` | 素体。`base_asian`（女性）か `base_male`（男性） |
| `BASE_SKIN` | 素体に合わせた肌 | Tripo が塗っていない所（耳の裏・体など）の肌。MPFB の `skins/` 以下のパス |
| `HEAD_ONLY` | （なし） | `1` で首で切った頭だけにする（体・服・靴を外す） |
| `PREP_PARTS` | 自動 | 顔に使う部品とテクスチャの置き場所（`部品番号:X0,Y0`）。書かなければ肌の色がいちばん多い部品 |
| `HAIR_PART` | `auto` | 髪の部品。`auto`（自動で見分ける）、`none`（髪なし）、部品番号 |
| `HAIR_SPRING` | `12` | 髪の揺れ方。数字は頭の周りに並べる骨の列の数、`0` で揺らさない、`updo:<セル>` はまとめ髪 |
| `ROOT_RGB` | 髪の色から | 髪の下の頭皮を塗る色（`R,G,B`） |
| `UP_RAISE` | `0` | 上まぶたの線を、中央で何 px 上げるか |
| `LIPS` | `paint` | `paint`：唇を Tripo の色で塗り直し、つやを足す。`keep`：Tripo のまま |
| `LASHES` | `painted` | `painted`：Tripo が描いたまつ毛のまま。`3d`：まつ毛を立体で生やす |
| `SLIM_GAIN` / `PORTRAIT` | `0` / — | `1.0` で、顔の下半分を元の肖像画（`PORTRAIT`）の輪郭へ寄せる（Tripo の顔が丸くなったとき） |
| `VRM_NAME` / `VRM_AUTHOR` | `NAME` / `unknown` | VRM のメタ情報の名前と作者 |

細かい調整（ふだんは触らなくてよいもの）：

| 名前 | 既定 | 意味 |
|---|---|---|
| `LIP_SYM` | `1` | 下唇の外側の線を左右対称にそろえる |
| `WHITE_ONLY` | `1` | 黒目の外側を、すべて Tripo の白目の平均色で塗る（白目に肌やまぶたの色が写るのを防ぐ） |
| `CORNEA` | `1` | 角膜をつやのある透明な材質にする（`0` で眼球と同じ材質） |
| `BLINK_BROW` | `1` | 目を閉じるとき眉を動かさない（眉が低い顔で、眉の毛がまぶたに伸びるのを防ぐ） |

`corner_adjust.txt`（任意、作業フォルダに置く）：1 行に `目 角 DX DY` と書きます。目は 0 が画面の左の目、角は `inner` か `outer`、DX・DY は確認画像の px です。例：`0 inner -6 -13`。

環境変数：`BLENDER`（Blender の実行ファイル）、`MPFB_DATA`（MPFB の素材フォルダ。既定は `~/Library/Application Support/Blender/5.2/extensions/.user/blender_org/mpfb/data`）。

## 既知の癖

- 「あ」「お」の口で、舌が前に出て見えることがあります。
- 首の継ぎ目で、顔の色と体の素体の肌の色が少し違うことがあります。
- 閉じたまぶたに、Tripo が描いたまぶたのしわが少し伸びて見えます。
- 角膜の透明な層は glTF の透過（`KHR_materials_transmission`）を使います。対応していないアプリでは、目が白っぽく見える可能性があります（`CORNEA=0` で外せます）。
- macOS でしか確かめていません（`md5 -q`、Blender と素材の既定の場所）。

## 中身

```
make_face.sh        全体の手順（部品の見分け → 形の合わせ → 色の仕上げ → 髪 → VRM の書き出し）
check.sh            事前チェック
setup.sh            仮想環境・モデルの取得・素体の画像の付け替え
face.env.example    設定の見本
viewer/             確認用のビューア（three.js + three-vrm）
tools/tripo/        Tripo の頭への合わせ（run.sh が順に呼ぶ）、まぶた・口・鼻、色の焼き付け、目の投影、唇、まばたきの直し
tools/tripo/hair/   髪の部品の見分け・移植・揺れ・頭皮の塗り
tools/mpfb/         素体（base_asian/・base_male/：ARKit の 52 キー入りの体と、顔の特徴点の対応）、合わせの適用、VRM の書き出し
tools/head_only.py  首で切ってふたをし、体・服・靴を外す（HEAD_ONLY=1）
.claude/skills/make-face/   Claude Code で作業するときの手順と確認の決まり
```

素体を作り直すための `tools/mpfb/build.py`（MPFB で素体を組む）、`add_arkit.py`（ARKit の 52 キーを足す）、`corr.py`（特徴点の対応）も入れてありますが、ふだんの手順では使いません。

## ライセンス

このリポジトリのスクリプトは [MIT License](LICENSE) です。使っているもの：

- **MPFB**（GPL-3.0）：同梱していません。Blender の拡張機能として各自で入れます。`tools/mpfb/` の一部のスクリプト（`build.py`・`add_arkit.py`・`settex.py`）は MPFB の API を呼びます。
- **MakeHuman system assets**（CC0）：同梱していません。`tools/mpfb/base_asian/`・`base_male/` の素体は、これから作ったものです。
- **MediaPipe Face Landmarker**（Apache-2.0）：`setup.sh` が取得します。
- **VRM format アドオン**（MIT）、**three.js**（MIT）、**three-vrm**（MIT）：同梱していません。
- 入力の Tripo のモデルと、そこから作った VRM の扱いは、Tripo の利用規約に従ってください。
