# セットアップガイド

## 前提条件

- Python 3.10+
- macOS (MPS) または CUDA GPU
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ
- Git

## 1. 環境構築

```bash
git clone git@github.com:Akatuki25/copillust.git
cd copillust

# Python 仮想環境
uv venv .venv --python 3.12
source .venv/bin/activate

# パッケージインストール
uv pip install -e .
```

### OpenMMLab スタック

OpenMMLab パッケージはビルドに特殊な対応が必要なため、uv pip で個別にインストールする。

```bash
# setuptools のバージョン制約 (pkg_resources 互換性)
uv pip install "setuptools<82"

# mmcv (ビルドに時間がかかる)
uv pip install mmcv==2.1.0 --no-build-isolation

# mmengine, mmdet
uv pip install mmengine==0.10.7 mmdet==3.3.0 --no-build-isolation

# xtcocotools (cython が必要)
uv pip install cython
uv pip install xtcocotools --no-build-isolation
```

### vendor/mmpose

mmpose はカスタムパッチ（MPS float64 修正、torch.load 互換性）を含むため、vendor ディレクトリにクローンして editable install する。

```bash
mkdir -p vendor
git clone --depth 1 --branch v1.3.2 https://github.com/open-mmlab/mmpose.git vendor/mmpose

# .mim symlink (mmpose がパッケージデータを見つけるために必要)
ln -sf $(pwd)/vendor/mmpose vendor/mmpose/mmpose/.mim

# pip 経由で editable install (uv pip は mmpose のビルドに非対応)
uv pip install pip
pip install -e vendor/mmpose/

# albumentations (Albumentation transform 用)
uv pip install "albumentations<2.0"
```

### MPS 向けパッチ

vendor/mmpose に以下のパッチが必要:

**1. RTMCCHead float64 修正** (`vendor/mmpose/mmpose/models/heads/coord_cls_heads/rtmcc_head.py` 行291):
```python
# Before:
acc_pose = torch.tensor(avg_acc, device=gt_x.device)
# After:
acc_pose = torch.tensor(avg_acc, device=gt_x.device, dtype=torch.float32)
```

**2. HeatmapHead float64 修正** (`vendor/mmpose/mmpose/models/heads/heatmap_heads/heatmap_head.py` 行314):
```python
# Before:
acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
# After:
acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device, dtype=torch.float32)
```

**3. torch.load 互換性** (`vendor/mmpose/tools/train.py` 先頭に追加):
```python
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
```

### SimCCLabel v1_weight パッチ

visibility 制御実験で使用。`vendor/mmpose/mmpose/codecs/simcc_label.py`:

- `__init__` に `v1_weight: Optional[float] = None` パラメータを追加
- `_map_coordinates` で v=1 の keypoint_weights を `self.v1_weight` で上書き

詳細は `pose_estimation/codecs/visibility_aware_simcc.py` のコメントを参照。

## 2. データセット準備

### Bizarre Pose

```bash
# リポジトリクローン
mkdir -p data/bizarre_pose
git clone --depth 1 https://github.com/ShuhongChen/bizarre-pose-estimator.git data/bizarre_pose/repo

# データセットダウンロード (repo の README に従う)
# bizarre_pose_dataset.zip を data/bizarre_pose/raw/ に展開

# COCO17 形式に変換
python -m pose_estimation.data.converters.bizarre_pose \
    --input data/bizarre_pose/raw/bizarre_pose_dataset/raw/annotations.json \
    --images data/bizarre_pose/raw/bizarre_pose_dataset/raw/images \
    --splits data/bizarre_pose/repo/bizarre-pose-estimator \
    --output data/bizarre_pose/coco/
```

変換後の構造:
```
data/bizarre_pose/
  coco/
    train.json          # 3200 images, COCO17 format
    val.json            # 313 images
    test.json           # 487 images
    train_visibility.json  # visibility 手動修正版
  raw/
    bizarre_pose_dataset/raw/images/  # 4000 PNG files
```

### Visibility アノテーション

Bizarre Pose の元データは全キーポイントが v=2 (visible)。手動修正ツールで v=0/1/2 を正しくアノテーション済み。

```bash
# visibility アノテーションツール起動
python scripts/tools/annotate_visibility.py --split train

# 進捗確認
cat data/bizarre_pose/coco/visibility_progress.json | python -c "import sys,json; print(json.load(sys.stdin)['reviewed'].__len__())"
```

### 学習用マージデータセット

```bash
# 基本マージデータ (Bizarre Pose train のみ)
mkdir -p data/merged/annotations data/merged/images
ln -sf $(pwd)/data/bizarre_pose/raw/bizarre_pose_dataset/raw/images data/merged/images/bizarre_pose

# アノテーションコピー (visibility 修正版の場合)
cp data/bizarre_pose/coco/train_visibility.json data/merged/annotations/train.json
cp data/bizarre_pose/coco/val.json data/merged/annotations/val.json

# train.json 内の file_name に "bizarre_pose/" prefix を追加する必要あり
# (merge スクリプトが自動で行う)
```

### Curriculum S1 用クリーンデータセット

```bash
mkdir -p data/merged_curriculum_s1/annotations data/merged_curriculum_s1/images
mkdir -p data/merged_curriculum_s1/images/bizarre_pose
ln -sf $(pwd)/data/bizarre_pose/raw/bizarre_pose_dataset/raw/images \
       data/merged_curriculum_s1/images/bizarre_pose/images

# v1<=1 の画像のみ抽出した annotations を作成
# (scripts/data/merge_stage_c.py 参照、または手動で visibility_progress.json からフィルタ)
```

## 3. 学習の実行

### Curriculum S2 (現在の best model) の再現

```bash
# Step 1: HumanArt pretrained → clean data (891枚) で 10ep
python -m pose_estimation.training.trainer \
    --config pose_estimation/models/configs/experiments/curriculum/humanart_curriculum_s1.py \
    --work-dir experiments/train/curriculum_s1 \
    --device mps

# Step 2: S1 checkpoint → 全 3200枚 (corrected visibility) で 10ep
python -m pose_estimation.training.trainer \
    --config pose_estimation/models/configs/experiments/curriculum/humanart_curriculum_s2.py \
    --work-dir experiments/train/curriculum_s2 \
    --device mps
```

S2 の config は S1 の best checkpoint を `load_from` で参照するため、S1 の完了が前提。

### 他のモデルの学習

```bash
# HRNet-W48 (COCO pretrained → BP fine-tune)
python -m pose_estimation.training.trainer \
    --config pose_estimation/models/configs/models/hrnet_w48_bizarre_pose.py \
    --work-dir experiments/train/hrnet_w48_bizarre_pose \
    --device mps
```

## 4. 評価

### ベンチマーク実行

```bash
# 利用可能なモデル一覧
python scripts/eval/benchmark.py --list

# Bizarre Pose test set で評価 (論文互換 OKS)
python scripts/eval/benchmark.py --models "Curriculum S2" --dataset bp

# mydata でも評価
python scripts/eval/benchmark.py --models "Curriculum S2" --dataset both
```

### モデル比較画像の生成

```bash
python scripts/eval/compare.py --left "Stage A" --right "Curriculum S2"
# → experiments/eval/comparisons/stagea_vs_curriculums2/
```

### 評価指標について

本プロジェクトでは **Bizarre Pose 論文 (WACV 2022) の評価方法** を使用:
- **OKS@50**: 個々のキーポイントの OKS が 0.5 以上の割合
- **OKS@75**: 個々のキーポイントの OKS が 0.75 以上の割合
- COCO sigmas 使用、bbox は GT keypoints から算出
- **COCO AP (pycocotools) とは異なる指標**。直接比較不可

## トラブルシューティング

### mmcv のビルドが失敗する

`pkg_resources` エラーの場合:
```bash
uv pip install "setuptools<82"
uv pip install mmcv==2.1.0 --no-build-isolation
```

### MPS で float64 エラー

vendor/mmpose のパッチ (上記) を適用済みか確認。

### UniFormer モデルが動かない

`projects.uniformer.models` の import に PYTHONPATH が必要:
```bash
PYTHONPATH=vendor/mmpose:$PYTHONPATH python -m pose_estimation.training.trainer ...
```
trainer.py は自動で vendor/mmpose を PYTHONPATH に追加する。
