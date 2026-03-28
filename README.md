# Copillust - Illustration Pose Estimation

イラスト（anime/manga/illustration）ドメインに特化したキャラクターポーズ推定。

## 現状

### ベンチマーク

Bizarre Pose test set (487枚) での論文互換評価 (per-keypoint OKS correctness rate):

| Model | OKS@50 | OKS@75 | params |
|-------|--------|--------|--------|
| 論文 best (Chen & Zwicker, WACV 2022) | 0.898 | 0.793 | 86.8M |
| **Curriculum S2 (ours)** | **0.892** | **0.801** | **14M** |
| RTMPose-m HumanArt (pretrained, no fine-tune) | 0.831 | 0.699 | 14M |
| RTMPose-m COCO (pretrained, no fine-tune) | 0.696 | 0.547 | 14M |

- OKS@75 で論文 best を上回る (0.801 vs 0.793)
- パラメータ数は論文の 1/6

### 技術選定

| 要素 | 選定 | 理由 |
|------|------|------|
| Base model | RTMPose-m (CSPNeXt + SimCC + GAU) | 軽量 (14M params) で HumanArt pretrained の効果が大きい |
| Pretrained | HumanArt (CVPR 2023) | COCO pretrained (+0.16 OKS@50)、イラスト含むドメインで事前学習 |
| Training data | Bizarre Pose 3200枚 (visibility 手動修正済み) | イラスト特化唯一のベンチマーク |
| Training strategy | Curriculum learning (clean → full) | clean data で基礎固め → occluded data で拡張 |
| Evaluation | 論文互換 OKS@50/75 on BP test | Chen & Zwicker (WACV 2022) と直接比較可能 |
| Framework | MMPose (OpenMMLab) | RTMPose の公式実装、学習・評価パイプライン完備 |

### 確認済みの知見

- **backbone の差し替えは効果なし**: HRNet-W48, UniFormer-B, ViTPose-B 全て Curriculum S2 以下。3200枚ではモデルサイズの恩恵がない
- **データセットの質が最大のレバー**: visibility アノテーション修正 + curriculum で best を達成。Amateur Drawings (178K枚の子供の落書き) の追加は全体悪化
- **写真ドメインの pretrained はイラストに転移しない**: DINOv2 の spectral decomposition でイラストのパーツ分離不可
- **occluded keypoint の扱いが精度のボトルネック**: 全て v=2 で学習していた Bizarre Pose の visibility を手動修正 (15.2% → v=1)

## 今後の方向性

[docs/model_modify.md](docs/model_modify.md) および [docs/model_modify_task.md](docs/model_modify_task.md) に詳細。

要約: **RTMPose のどの仮定がイラストで壊れているか** を特定し、visibility / relation / presence のどれが remaining error を支配しているかを切り分ける。

### 第一ボトルネック: Supervision semantics (visibility)

occluded keypoint を座標教師としてどの強さで学習させるかの制御。masking (feature / attention) ではなく loss weight 側での制御が有望。

### 第二ボトルネック: Relation prior

GAU の関節関係が COCO/HumanArt 寄りで anime 体型に硬い。デフォルメ・極端ポーズでの破綻。

### 第三ボトルネック: Tail failure

低品質な予測（confidence threshold 実験で OKS@50 が 0.892→0.934 に改善）。ProbPose (CVPR 2025) の presence / visibility 分離が参考。

## プロジェクト構造

```
pose_estimation/
  models/
    configs/
      base/                    # 共通設定 (データ, パイプライン, スケジュール)
      models/                  # モデルアーキテクチャ定義
      experiments/
        stages/                # 初期段階実験 (A-D)
        curriculum/            # カリキュラム学習
        visibility/            # visibility 効果検証
        occluded_reasoning/    # Occluded Reasoning (進行中)
    heads/
      occluded_rtmcc_head.py   # visibility-aware RTMCCHead
    losses/
      structural_loss.py       # 骨格制約 loss (実験的)
  codecs/
    visibility_aware_simcc.py  # v1_weight 制御付き SimCC codec
  transforms/
    random_edges_blackout.py   # ProbPose 由来の crop augmentation
    visibility_weight_control.py
  training/
    trainer.py                 # MMPose train.py wrapper
  core/                        # 定数, bbox 等
  data/                        # データ変換 (Bizarre Pose → COCO)

scripts/
  eval/
    model_registry.py          # 全モデルの定義 (1箇所管理)
    metrics.py                 # 評価関数 (論文互換 OKS)
    benchmark.py               # ベンチマーク実行
    compare.py                 # 2モデル比較画像生成
  tools/
    annotate_keypoints.py      # GT キーポイント アノテーションツール
    annotate_visibility.py     # Visibility アノテーションツール

docs/
  model_modify.md              # モデル改善方針
  model_modify_task.md         # 4仮定の分析と優先順位
  dataset.md                   # データセット情報
```

## セットアップ

```bash
# 環境構築
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv pip install --no-build-isolation mmcv mmengine mmdet
pip install -e vendor/mmpose/

# ベンチマーク実行
python scripts/eval/benchmark.py --list
python scripts/eval/benchmark.py --models "Curriculum S2" --dataset bp
```

## データセット

- **Bizarre Pose**: 4000枚の Danbooru イラスト (3200 train / 313 val / 487 test)
  - COCO17 keypoints, 全件 visibility 手動修正済み
  - [ShuhongChen/bizarre-pose-estimator](https://github.com/ShuhongChen/bizarre-pose-estimator)
- **mydata**: 18枚の手動 GT アノテーション付きイラスト (開発検証用、ベンチマークではない)

## 参考論文

- Chen & Zwicker, "Transfer Learning for Pose Estimation of Illustrated Characters", WACV 2022
- Ju et al., "Human-Art: A Versatile Human-Centric Dataset", CVPR 2023
- Jiang et al., "RTMPose: Real-Time Multi-Person Pose Estimation", 2023
- Sun et al., "Rethinking Visibility in Human Pose Estimation", WACV 2024
- Purkrabek et al., "ProbPose: A Probabilistic Approach to 2D Human Pose Estimation", CVPR 2025
