# Windows (CUDA 16GB) での R2a/R2b 実行手順

Mac (M3 Pro) で検証済みのパイプラインを CUDA 環境に移す手順。
生成はネイティブ Windows で可、学習 (mmpose) は WSL2 推奨。

## 持っていくもの

| 対象 | 入手方法 | 備考 |
|------|-----------|------|
| コード・レンダ済み126シーン (`r2a/renders_v2/`, 89MB)・S2 checkpoint (52MB) | **git clone で完結** (autoexp/apr8 にコミット済み) | Blender 再実行不要。再レンダはバージョン差で画素が変わるため非推奨 (凍結入力) |
| `experiments/synth/assets/vrm/` | **CLI で再DL 可**: `https://github.com/madjin/vrm-samples/raw/master/` の `vroid/fem_vroid.vrm`, `vroid/masc_vroid.vrm`, `Seed-san/vrm/Seed-san.vrm` (→ `seed_san.vrm` にリネーム)。Mac と同一ファイル | ライセンス配慮で未コミット。pixiv 公式サンプル系 (改変・自由利用可の規約) だが再配布はしない運用 |
| HF モデル (NoobAI-XL / MistoLine / vae-fix) | 現地で自動ダウンロード (~10GB) | 初回実行時に hf_cache/ へ落ちる |

## 1. 生成環境 (ネイティブ Windows で可) — 2026-07-04 RTX 5060 Ti で構築済み

uv で構築 (venv より速い)。RTX 5060 Ti は Blackwell (sm_120) のため
**cu124 wheel は不可 — cu128 (torch 2.7+) が必須**:

```powershell
cd experiments\synth
uv venv genv --python 3.12
uv pip install --python genv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python genv\Scripts\python.exe diffusers "transformers<5" accelerate safetensors pillow opencv-python-headless huggingface_hub
```

構築時の罠 (2026-07-04 に踏んだもの):
- **transformers は 5 系不可** (5.13.0 で NoobAI-XL の text_encoder_2 が
  デフォルト config で初期化され shape mismatch)。`transformers<5` (4.57.6 で動作確認)
- HF ダウンロードが不完全だと `unet/config.json` 欠落エラーになる。
  大きい safetensors を捨てずに `snapshot_download(..., allow_patterns=['*.json','*.txt','**/*.json','**/*.txt'])`
  で config だけ追加取得すれば復旧できる
- generate_sketch.py は CUDA 自動対応済み (コード変更不要)

実行 (残りマトリクス):

```powershell
genv\Scripts\python generate_sketch.py --styles rough,pencil,lineart --scales 0.55,0.9
```

シーン一覧は `r2a/renders_v2/` のディレクトリ名 (--scenes 省略で全シーン。
既存出力はスキップされるので再実行安全)。実測 ~13秒/枚 (5060 Ti, 768×1024, 28 steps)
→ 全126シーン×3スタイル×2スケール = 756枚で約3時間。

## 2. 測定 (S2 再推定ゲート) — ネイティブ Windows + CPU 推論で可 (2026-07-04 確立)

WSL2 は不要。推論のみなら CPU で数分 (756枚 ≈ 2-3分)。リポジトリルートに
測定用 venv を作る (mmcv は OpenMMLab の Windows CPU wheel でビルド不要):

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe mmcv==2.1.0 --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html "numpy<2" mmengine==0.10.7 mmdet==3.3.0
uv pip install --python .venv\Scripts\python.exe "setuptools<82" cython pip xtcocotools pycocotools tqdm pandas json-tricks scipy munkres
uv pip install --python .venv\Scripts\python.exe -e . --no-deps
git clone --depth 1 --branch v1.3.2 https://github.com/open-mmlab/mmpose.git vendor/mmpose
New-Item -ItemType Junction -Path vendor\mmpose\mmpose\.mim -Target (Resolve-Path vendor\mmpose)
.venv\Scripts\python.exe -m pip install -e vendor\mmpose --no-build-isolation --no-deps
```

(SETUP.md の MPS float64 / v1_weight パッチは学習用 — S2 の CPU 推論には不要)

```powershell
.venv\Scripts\python.exe experiments\synth\measure_fidelity.py --gen experiments\synth\r2a\gen --renders experiments\synth\r2a\renders_v3 --model "Curriculum S2" --also-line --out experiments\synth\r2a\fidelity_v3.json
```

注意: measure_fidelity は --gen 配下の全シーンを --renders の同名 gt と突き合わせる。
**条件レンダの世代 (v2/v3) と生成物のペアを混ぜないこと** (junction で部分集合を
作ってフィルタする)。

### v3 測定結果 (2026-07-04, fem/masc 84シーン×6条件 + 対照)

| 条件 | PCK@0.1 | PCK@0.2 | 備考 |
|------|---------|---------|------|
| line (対照=手続きレンダ) | 0.954 | 0.996 | normal 0.995 / chibi 0.913 |
| cs0.90 (rough/pencil/lineart) | 0.940-0.946 | 0.987-0.988 | **対照とほぼ同等 = ラベル忠実度維持** |
| cs0.55 (同) | 0.856-0.866 | 0.919-0.947 | スタイルは出るが約-0.09 のドリフト |

- チェッカー (S2) 自体のスケッチ弱さ込みの下限値。chibi は normal より一貫して低い
- seed_san×v2 の再測定で、v2 の低スコアは bust カメラ破綻由来と確認
  (line 対照自体が bust で 0.40; full/high では cs0.9 ≈ 対照)。fidelity_seed_v2.json

## 3. R2b: 混合再学習 (2026-07-04 パイプライン確立)

### 生成構成 (ベンチ確定)

| 構成 | 速度 (5060 Ti) | 忠実度 PCK@0.1 (32枚ベンチ) |
|------|------|------|
| steps28 / batch1 (R2a 従来) | 11.9s/枚 | 0.829 |
| **steps16 / batch1 (R2b 採用)** | **6.9s/枚** | **0.884** |
| batch≥2 | 8.4〜10.2s/枚 | (VRAM 溢れで逆効果 — batch は 1 固定) |

steps 削減は速度 1.7倍 + 忠実度も改善 (デノイズ工程が短いほど条件から離れない)。

### データ生成 (Windows ネイティブ)

```powershell
# 1) レンダスイープ: 64 シード × 126 シーン構成 = 8,064 シーン (~2.5h, 4並列)
powershell -File render_r2b.ps1   # --jscale 2 --passes line --seed-in-name

# 2) 生成: シーンごとに (style,cs) を2条件サンプル = 16,128枚 (~31h)
$env:RENDERS_DIR = "...\r2b\renders"; $env:GEN_OUT = "...\r2b\gen"
genv\Scripts\python generate_sketch.py --variants 2 --steps 16 --batch 1

# 3) 品質ゲート測定 → COCO 変換 → BP と混合
.venv\Scripts\python experiments\synth\measure_fidelity.py --gen ...\r2b\gen --renders ...\r2b\renders --model "Curriculum S2" --out ...\r2b\fidelity.json
genv\Scripts\python synth_to_coco.py --gen r2b\gen --renders r2b\renders --out r2b\synth_train.json --fidelity r2b\fidelity.json --min-pck02 0.75
genv\Scripts\python merge_r2b.py --bp <data>/merged/annotations/train.json --synth r2b\synth_train.json --out <data>/merged/annotations/train_r2b.json
```

### 学習 (WSL2 — 環境構築済み: ~/copillust, torch cu128 + mmcv CPU-ops ソースビルド)

WSL 側に r2b/gen をコピーし (9P 越え I/O は遅い)、
`data/merged/images/synth -> r2b/gen` を symlink した上で:

```bash
# 本命: BP 3200 + synth 混合 (config: experiments/synth/r2b_mixed.py)
.venv/bin/python -m pose_estimation.training.trainer \
  --config pose_estimation/models/configs/experiments/synth/r2b_mixed.py \
  --work-dir experiments/train/r2b_mixed --device cuda
# 対照: 同一スケジュールで BP のみ (synth 効果とエポック追加効果の分離)
.venv/bin/python -m pose_estimation.training.trainer \
  --config pose_estimation/models/configs/experiments/synth/r2b_bp_only.py \
  --work-dir experiments/train/r2b_bp_only --device cuda
```

判定: BP test / Crop-Bizarre / **mydata lineart+chibi OKS@50 で S2 比 +0.05**
(redesign_proposal §7)。mydata 画像は gitignore のため要転送。
WSL 学習実測: 3200枚/エポック ≈ 20秒 (batch32) — 19K 混合でも数分/エポック。
ドライラン時の注意: S2 config の評価器は `data/merged_500_corrected/annotations/val.json`
を参照する (data/merged の val.json をコピーで可)。

## 2026-07-04 改訂: renders_v2 の欠陥と修正 (Windows 側)

全126シーンのコンタクトシート (`r2a/sheet_line_{chibi,normal}.png`) 監査で判明:

1. **ポーズ適用バグ**: 旧 `aim_bone` は head→tail 方向基準だったが、glTF に tail は
   存在せずインポータの推測値のため fem/masc リグで半端な回転になっていた
   (armup が水平、walk で足が顔の前など)。→ head→**子関節の実座標** 基準に修正済み
2. **カメラの空振り**: bust/lie が「無地の胴のアップ」になっていた。
   → キーポイント bbox を画角にフィットさせる方式に修正済み。体が水平 (lie) なら
   カメラを持ち上げる
3. **方針(b) 採用**: マネキン (fem/masc vroid) シーンは生成プロンプトに髪・服タグを
   ランダム付与して「ちゃんとしたキャラ」を描かせる (実スケッチの分布に寄せる)。
   seed_san は条件線画に髪・服が既にあるため付与しない。
   既知の代償: 発明された髪・服による遮蔽は gt.json の v フラグに反映されない
4. **素体の結論 (試行錯誤の末)**: Blender VRM アドオンの `icyp.make_basic_armature` +
   Skin モディファイアで素体を CLI 生成する案は**不採用** (人体形状から逸脱しすぎ、
   拡散条件として品質リスク — make_parametric_vrm.py ヘッダ参照)。
   ベース素体は **VRoid Studio エクスポートの人型 VRM** を使う。chibi は
   体型スライダーによるネイティブ頭身版を `*_chibi.vrm` の名前で出せば、
   blender_render.py が骨スケール無しでそのまま扱う (ファイル名 build 対応済み)。
   従来の骨スケール chibi も人間形状は保つため暫定利用可

再レンダ手順 (fem/masc 全シーン。**renders_v2 は凍結のまま、v3 を新設**):

```powershell
blender -b -P blender_render.py -- --vrm assets\vrm\fem_vroid.vrm --out r2a\renders_v3 --poses stand,armup,sit,walk,wave,crouch,lie --builds normal,chibi --cams full,bust,high
# masc_vroid.vrm も同様。assets/vrm は未コミット (ライセンス) — Mac からコピー
```

注意: プロンプト変更により、修正前に生成済みの fem/masc の `r2a/gen/<scene>/` は
旧プロンプト産。v3 レンダで生成し直す前に該当ディレクトリを削除すること
(スキップ判定はファイル存在ベースのため混在する)。

## Mac で確定済みの技術ノート (再現時の罠)

1. `enable_attention_slicing()` の NaN は MPS 固有 — CUDA では使わない (不要)
2. VAE は `madebyollin/sdxl-vae-fp16-fix` を使用 (SDXL fp16 の既知 NaN 回避)
3. NoobAI-XL 1.1 は eps-pred。EulerAncestral + cfg 5.5 / 28 steps
4. MistoLine への条件は白線・黒背景 (line.png を反転して入力)
5. 品質ゲート: 生成後に S2 再推定 PCK@0.2 < 1.0 のサンプルは要目視 (現状の
   閾値は仮。規模実行時に ROC を見て決める)
