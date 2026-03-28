# アーキテクチャドキュメント

## ディレクトリ構造と役割

```
copillust/
├── pose_estimation/          # メインパッケージ
│   ├── core/                 # 基盤クラス・定数
│   ├── models/               # モデル定義
│   │   ├── configs/          # MMPose 設定ファイル群
│   │   ├── heads/            # カスタム head 実装
│   │   └── losses/           # カスタム loss 実装
│   ├── codecs/               # カスタム SimCC codec
│   ├── transforms/           # カスタム data augmentation
│   ├── training/             # 学習パイプライン
│   ├── inference/            # 推論パイプライン
│   ├── evaluation/           # 評価モジュール
│   └── data/                 # データ変換・マージ
├── scripts/                  # CLI ツール群
│   ├── eval/                 # 評価・ベンチマーク
│   ├── tools/                # アノテーションツール
│   └── data/                 # データ準備
├── docs/                     # ドキュメント
├── vendor/                   # 外部依存 (gitignore)
│   └── mmpose/               # パッチ済み MMPose v1.3.2
├── data/                     # データセット (gitignore)
├── experiments/              # 学習結果 (gitignore)
└── mydata/                   # テスト用イラスト (画像は gitignore)
```

## 各ディレクトリの詳細

### pose_estimation/core/

プロジェクト全体で使う基盤。

| ファイル | 役割 |
|---------|------|
| `constants.py` | COCO17 キーポイント名、スケルトン定義、インデックスマッピング |
| `types.py` | `PoseResult`, `KeypointResult`, `BoundingBox` 等の型定義 |
| `base_estimator.py` | 推定器の抽象基底クラス |
| `base_evaluator.py` | 評価器の抽象基底クラス |

### pose_estimation/models/

#### rtmpose_estimator.py

MMPose の `init_model` + `inference_topdown` をラップした推論クラス。

```
入力画像 → init_model(config, checkpoint) → inference_topdown(model, image, bboxes)
         → PoseDataSample → PoseResult(keypoints=[KeypointResult(x, y, confidence)])
```

- bboxes=None の場合は画像全体を1人の bbox として使用
- MPS デバイスの場合 CPU にフォールバック（MMPose の MPS 対応が不完全なため）
- torch.load の weights_only パッチを内包

#### configs/base/base_bizarre_pose.py

全実験で共有される設定。新しい実験 config はこれを `_base_` で継承し、差分だけ記述する。

定義内容:
- Runtime: default_scope, hooks, env_cfg, visualizer
- Dataset: CocoDataset, data_root, ann_file paths
- Pipeline: LoadImage → GetBBoxCenterScale → RandomFlip → RandomBBoxTransform → TopdownAffine → GenerateTarget → PackPoseInputs
- Dataloader: batch_size=32, num_workers=2
- Schedule: 10 epochs, AdamW lr=5e-4, LinearLR warmup + CosineAnnealingLR

#### configs/models/

各モデルアーキテクチャの定義。base config を継承し、`model` と `codec` だけ override する。

| ファイル | backbone | head | pretrained |
|---------|----------|------|-----------|
| rtmpose_m_humanart_pretrained.py | CSPNeXt-m | RTMCCHead (SimCC) | HumanArt |
| rtmpose_l_humanart.py | CSPNeXt-l | RTMCCHead (SimCC) | HumanArt |
| hrnet_w48_bizarre_pose.py | HRNet-W48 | HeatmapHead | COCO |
| hrnet_w48_dark_bizarre_pose.py | HRNet-W48 | HeatmapHead+DARK | COCO |
| uniformer_b_bizarre_pose.py | UniFormer-B | HeatmapHead | COCO |
| vitpose_base_bizarre_pose.py | ViT-B | HeatmapHead+UDP | COCO |

#### configs/experiments/

実験ごとの config。models/ の config またはbase config を継承し、学習戦略を override する。

```
experiments/
├── stages/               # Stage A-D: 初期実験
│   ├── rtmpose_m_stage_a.py    # COCO pretrained → BP 10ep (baseline)
│   ├── rtmpose_m_stage_b.py    # Stage A → 50ep + 強 augmentation
│   ├── rtmpose_m_stage_c.py    # + Amateur Drawings 178K (失敗)
│   └── rtmpose_m_stage_d.py    # 384x288 + 複数条件同時変更 (失敗)
├── curriculum/           # カリキュラム学習
│   ├── humanart_curriculum_s1.py   # HumanArt → clean 891枚で 10ep
│   └── humanart_curriculum_s2.py   # S1 → 全 3200枚 corrected で 10ep (best)
├── visibility/           # Visibility アノテーション効果
│   ├── humanart_500_corrected.py   # corrected v=0/1/2
│   └── humanart_500_allv2.py       # 同じ画像、全て v=2
├── occluded_reasoning/   # Occluded Reasoning (進行中)
│   ├── occluded_reasoning.py       # 1層 GAU + hard masking
│   ├── occluded_3layer.py          # 3層 GAU + hard masking
│   ├── gau_3layer_no_mask.py       # 3層 GAU masking なし
│   ├── soft_attn_curriculum_*.py   # soft attention masking
│   ├── occ3l_curriculum_*.py       # 3層 GAU + curriculum
│   ├── p1_crop_augment.py          # RandomEdgesBlackout 単体
│   └── v1_visible_only_s1.py       # visible-only 学習 (v=1 除外)
└── _deprecated/          # 効果なしの実験
```

#### heads/occluded_rtmcc_head.py

RTMCCHead を拡張した OccludedRTMCCHead。WACV 2024 の Occluded Pose Reasoning に基づく。

```
backbone features → final_conv [B,K,H,W] → flatten [B,K,HW]
                          |
                    VisNet (MLP) → pred_vis [B,K]    # visibility 予測
                          |
                    Masking (hard or soft)              # occluded の特徴を減衰
                          |
                    MLP → [B,K,hidden]
                          |
                    GAU × N layers (self-attention)    # keypoint 間 reasoning
                          |
                    SimCC cls_x, cls_y                 # 座標分類
```

パラメータ:
- `num_gau_layers`: GAU の層数 (1 or 3)
- `occ_attenuation`: masking の減衰率 (0.01=hard, 0.3=soft, 1.0=なし)
- `soft_attention_mask`: True の場合、feature ではなく attention kernel に masking
- `vis_loss_weight`: visibility prediction BCE loss の重み
- `mix_gt_prob`: 学習時に GT visibility を使う確率 (train-test gap 対策)

### pose_estimation/codecs/

#### visibility_aware_simcc.py

SimCCLabel のサブクラス。`v1_weight` パラメータで occluded keypoint の loss weight を制御。

```python
# 標準 SimCCLabel: v=0→weight=0, v=1→weight=1, v=2→weight=2
# VisibilityAwareSimCC(v1_weight=0.0): v=1→weight=0 (visible-only training)
# VisibilityAwareSimCC(v1_weight=0.5): v=1→weight=0.5 (半分の重み)
```

注: vendor/mmpose の SimCCLabel 本体にも `v1_weight` パッチを適用済み。

### pose_estimation/transforms/

| ファイル | 役割 | pipeline 内の位置 |
|---------|------|-----------------|
| `random_edges_blackout.py` | 画像の端をランダムに黒塗り。ProbPose (CVPR 2025) 由来 | TopdownAffine の後 |
| `visibility_weight_control.py` | GenerateTarget 後に keypoint_weights を visibility に応じて上書き | GenerateTarget の後 |

### pose_estimation/training/trainer.py

MMPose の `vendor/mmpose/tools/train.py` を subprocess で呼び出すラッパー。

```
trainer.py --config CONFIG --work-dir DIR --device DEVICE
    → subprocess: python vendor/mmpose/tools/train.py CONFIG --work-dir DIR --cfg-options device=DEVICE
```

- MPS の場合 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` を環境変数に設定
- vendor/mmpose を PYTHONPATH に追加（UniFormer 等の projects/ import 用）

### pose_estimation/data/

| ファイル | 役割 |
|---------|------|
| `converters/bizarre_pose.py` | Bizarre Pose 独自形式 (25-joint, [y,x] order) → COCO17 JSON |
| `converters/amateur_drawings.py` | Amateur Drawings 16-joint → COCO17 (使用せず) |
| `converters/humanart.py` | Human-Art 変換ユーティリティ |
| `coco_utils.py` | COCO JSON の load/save、category 生成 |
| `merge.py` | 複数データセットのマージ（sampling ratio 付き） |
| `download.py` | データセットダウンロード手順の表示 |

### scripts/eval/

ベンチマーク・評価システム。

#### model_registry.py

全モデルの config パスと checkpoint パスを 1 箇所で管理。benchmark.py と compare.py が参照。

```python
ALL_MODELS = {
    "Stage A": (config_path, checkpoint_path),
    "Curriculum S2": (...),
    ...
}
```

新しいモデルを追加する場合はここだけ編集。

#### metrics.py

Bizarre Pose 論文互換の OKS 評価関数。

```python
paper_oks_per_image(gt_kps, pred_kps, bbox, thresh=0.5)
# → fraction of visible keypoints with per-keypoint OKS >= threshold
```

- COCO sigmas 使用
- bbox の max dimension で座標を正規化（論文の方法に準拠）
- **COCO AP (pycocotools) とは異なる**

#### benchmark.py

```bash
python scripts/eval/benchmark.py --models "Curriculum S2" "Stage A" --dataset bp
```

model_registry から指定モデルを順に読み込み、BP test / mydata で OKS@50, OKS@75 を算出。論文のスコアも並べて表示。

#### compare.py

2 モデルの推論結果を並べた比較画像を生成。

```bash
python scripts/eval/compare.py --left "Stage A" --right "Curriculum S2"
```

### scripts/tools/

#### annotate_keypoints.py

OpenCV ベースの GUI で mydata にキーポイント GT を打つ。17 点を順番にクリック、右クリックで not visible。COCO JSON 形式で保存。

#### annotate_visibility.py

Bizarre Pose の既存キーポイント座標に対して visibility (v=0/1/2) を修正する GUI。画像にキーポイントを色付き表示し、クリックで visibility を切り替え。50 枚ごとに自動保存、レジューム対応。

## データフロー

### 学習

```
[Bizarre Pose raw annotations]
    ↓ converters/bizarre_pose.py
[COCO17 JSON] (data/bizarre_pose/coco/train.json)
    ↓ annotate_visibility.py (手動)
[Visibility 修正版] (data/bizarre_pose/coco/train_visibility.json)
    ↓ merge / symlink
[merged dataset] (data/merged/ or data/merged_curriculum_s1/)
    ↓
[MMPose training pipeline]
    LoadImage → GetBBoxCenterScale → RandomFlip → RandomBBoxTransform
    → TopdownAffine(192x256) → GenerateTarget(SimCCLabel) → PackPoseInputs
    ↓
[RTMPose-m model]
    CSPNeXt backbone → final_conv → flatten → MLP → GAU → cls_x, cls_y
    ↓
[KLDiscretLoss] (target_weight = visibility value)
    ↓
[Checkpoint] (experiments/train/*/best_coco_AP_epoch_N.pth)
```

### 推論

```
[入力画像]
    ↓ RTMPoseEstimator.predict(image)
[init_model(config, checkpoint)]
    ↓ inference_topdown(model, image, bboxes=[[0,0,w,h]])
[PoseDataSample]
    ↓ keypoints, keypoint_scores を抽出
[PoseResult(keypoints=[KeypointResult(x, y, confidence) × 17])]
```

### 評価

```
[モデル checkpoint] + [テストデータ (BP test 487枚)]
    ↓ scripts/eval/benchmark.py
[各画像で推論 → pred_kps]
    ↓ metrics.paper_oks_per_image(gt_kps, pred_kps, bbox, thresh)
[per-keypoint OKS → correctness rate]
    ↓ 全画像の平均
[OKS@50, OKS@75]
    ↓ 論文スコアと並べて表示
```

## RTMPose のアーキテクチャ詳細

本プロジェクトで使用する RTMPose-m の内部構造:

```
入力画像 [B, 3, 256, 192]
    ↓
CSPNeXt-m backbone (deepen=0.67, widen=0.75)
    ↓ stride=32 で downsample
feature map [B, 768, 8, 6]
    ↓
final_layer: Conv2d(768→17, kernel=7×7, padding=3)
    ↓
[B, 17, 8, 6]  (17 keypoints × spatial)
    ↓ flatten
[B, 17, 48]
    ↓
MLP: ScaleNorm → Linear(48→256)
    ↓
[B, 17, 256]  (17 keypoint tokens)
    ↓
GAU (Gated Attention Unit): 17 tokens の self-attention
    - Q, K: learned projection (dim=128)
    - Attention: squared ReLU (not softmax)
    - Gate: u * (attn @ v), expansion_factor=2
    - Residual + Scale
    ↓
[B, 17, 256]
    ↓
cls_x: Linear(256→384)  # 192 * split_ratio=2.0
cls_y: Linear(256→512)  # 256 * split_ratio=2.0
    ↓
pred_x [B, 17, 384], pred_y [B, 17, 512]
    ↓ argmax (推論時) or KLDiscretLoss (学習時)
keypoint coordinates
```

### SimCC の座標表現

- 入力 192×256 に対して split_ratio=2.0 で x=384 bins, y=512 bins
- 各 bin は 0.5 pixel に対応（量子化誤差 0.25px）
- 学習時: GT 座標に Gaussian (sigma=4.9, 5.66) を配置した 1D 分布がターゲット
- KLDiscretLoss の beta=10 が予測分布を鋭くする温度パラメータ

### GAU の inter-keypoint reasoning

GAU は 17 個の keypoint token 間の自己注意を計算:
- 「肩がここなら肘はこの辺」という関節間の空間的関係を学習
- squared ReLU attention (softmax ではない) で高速化
- 1 層のみ（標準 RTMPose）。本プロジェクトでは 3 層に拡張する実験も実施

### Visibility と loss weight の関係

mmpose の SimCCLabel codec:
```
keypoint_weights = keypoints_visible.copy()
```

- v=0 (未ラベル): weight=0 → loss に寄与しない
- v=1 (occluded): weight=1 → loss に寄与するが v=2 の半分の重み
- v=2 (visible): weight=2 → 最大重み

Bizarre Pose の元データは全て v=2。手動修正で 15.2% を v=1 に変更済み。
