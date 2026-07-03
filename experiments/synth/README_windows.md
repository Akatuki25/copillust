# Windows (CUDA 16GB) での R2a/R2b 実行手順

Mac (M3 Pro) で検証済みのパイプラインを CUDA 環境に移す手順。
生成はネイティブ Windows で可、学習 (mmpose) は WSL2 推奨。

## 持っていくもの

| 対象 | サイズ目安 | 備考 |
|------|-----------|------|
| `experiments/synth/*.py` + `README_windows.md` | 数十KB | リポジトリごと clone でよい |
| `experiments/synth/r2a/renders_v2/` (126シーン) | ~200MB | レンダ済みアセット。Blender 再実行不要 |
| `experiments/synth/assets/vrm/` | ~35MB | 追加レンダしたい場合のみ |
| HF モデル (NoobAI-XL / MistoLine / vae-fix) | ~10GB | コピーせず現地で再ダウンロード推奨 |

## 1. 生成環境 (ネイティブ Windows で可)

```powershell
cd experiments\synth
python -m venv genv
genv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
genv\Scripts\pip install diffusers transformers accelerate safetensors pillow opencv-python-headless huggingface_hub
```

generate_sketch.py の変更点 (CUDA 自動対応にするには):
- `device = "cuda" if torch.cuda.is_available() else ...`
- `dtype = torch.float16` に戻してよい (attention slicing NaN は **MPS 固有**。
  CUDA では fp16 + xformers/SDPA で問題なし。slicing 自体 16GB では不要)

実行 (残りマトリクス):

```powershell
genv\Scripts\python generate_sketch.py --scenes <scene1,scene2,...> --styles rough,pencil,lineart --scales 0.55,0.9
```

シーン一覧は `r2a/renders_v2/` のディレクトリ名。全126シーン×3スタイル×2スケール
= 756枚も 16GB CUDA なら現実的 (~8-16時間)。まず R2a 用の代表10シーン (60枚) から。

## 2. 測定 (S2 再推定ゲート) — WSL2 推奨

リポジトリのセットアップ (SETUP.md) を WSL2 内で行い、checkpoint
(`experiments/train/curriculum_s2/`) を配置した上で:

```bash
.venv/bin/python experiments/synth/measure_fidelity.py \
  --gen experiments/synth/r2a/gen --renders experiments/synth/r2a/renders_v2 \
  --model "Curriculum S2" --also-line --out experiments/synth/r2a/fidelity.json
```

比較シート: `genv/bin/python make_sheet.py --gen r2a/gen --renders r2a/renders_v2 --out r2a/sheets`

## 3. R2b: 混合再学習 (WSL2)

R2a 合格後。数万枚生成 → COCO 形式に変換 (gt.json は既に COCO17 順 + v flag) →
BP 3.2K と混合し、`humanart_curriculum_s2.py` ベースの config で fine-tune。
batch 256 / 192×256 で VRAM ~10GB。変換スクリプトは R2a 合格後に作成する。

## Mac で確定済みの技術ノート (再現時の罠)

1. `enable_attention_slicing()` の NaN は MPS 固有 — CUDA では使わない (不要)
2. VAE は `madebyollin/sdxl-vae-fp16-fix` を使用 (SDXL fp16 の既知 NaN 回避)
3. NoobAI-XL 1.1 は eps-pred。EulerAncestral + cfg 5.5 / 28 steps
4. MistoLine への条件は白線・黒背景 (line.png を反転して入力)
5. 品質ゲート: 生成後に S2 再推定 PCK@0.2 < 1.0 のサンプルは要目視 (現状の
   閾値は仮。規模実行時に ROC を見て決める)
