# autoexp

RTMPose イラストポーズ推定の自動実験ループ。[karpathy/autoresearch](https://github.com/karpathy/autoresearch) に倣った設計。

## 構成

```
autoexp/
  program.md         # Claude Code への指示書（ループ手順・制約・目標）
  evaluate.py        # 固定評価コード（触らない）
  results.tsv        # 実験ログ（git 追跡なし）
  hypotheses/
    bottleneck1_visibility.md  # 仮説空間: visibility supervision
    bottleneck2_relation.md    # 仮説空間: GAU relation prior
    bottleneck3_presence.md    # 仮説空間: tail failure / presence
```

## 対応する autoresearch の設計

| このプロジェクト | autoresearch |
|---|---|
| `autoexp/program.md` | `program.md` — エージェントへの指示書 |
| `autoexp/evaluate.py` | `prepare.py` — 固定評価、不変 |
| `pose_estimation/models/configs/experiments/autoexp/` + カスタム Python モジュール | `train.py` — エージェントが触るゾーン |
| `autoexp/results.tsv` | `results.tsv` — 実験ログ |

runner.py は存在しない。**Claude Code 自身がエージェント**として動く。

## 実行方法

### 前提

- 環境セットアップ済み（`source .venv/bin/activate`）
- Curriculum S2 チェックポイントが存在する（`experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth`）
- BP test データが存在する（`data/bizarre_pose/coco/test.json`）

### 自律実験ループを開始する

Claude Code のセッションを開いて、`autoexp/program.md` を読ませる：

```
このプロジェクトの autoexp/program.md に従って実験ループを開始してください。
```

または直接：

```bash
# Claude Code CLI から
claude "autoexp/program.md に従って autoexp/apr8 ブランチで実験ループを開始してください"
```

Claude Code が `program.md` の手順に沿って自律的に動く：
1. ブランチ作成 (`git checkout -b autoexp/<tag>`)
2. 対象ファイルを読む
3. `autoexp/results.tsv` を初期化
4. ベースライン評価（既存の Curriculum S2 checkpoint を評価）
5. 仮説を選んで config / Python モジュールを変更
6. `git commit`
7. 学習実行 → チェックポイント取得 → 評価
8. `results.tsv` に記録
9. OKS@75 改善 + mydata sanity pass → ブランチ前進
10. 改善なし → `git reset --hard HEAD~1`
11. ループ

### 評価だけ実行する（手動）

```bash
python -m autoexp.evaluate \
    --config pose_estimation/models/configs/experiments/curriculum/humanart_curriculum_s2.py \
    --checkpoint experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth
```

```
Evaluation Results
==================================================
Bizarre Pose test (487 images):
  OKS@50: 0.892
  OKS@75: 0.801  ← primary metric (baseline: 0.801)

mydata sanity (18 images):
  OKS@50: 0.xxx
  OKS@75: 0.xxx

Delta vs baseline: +0.000
```

オプション：

```bash
# GPU 使用
python -m autoexp.evaluate --config ... --checkpoint ... --device cuda

# mydata チェックをスキップ（高速）
python -m autoexp.evaluate --config ... --checkpoint ... --skip-mydata
```

### 学習だけ実行する（手動）

```bash
python -m pose_estimation.training.trainer \
    --config pose_estimation/models/configs/experiments/autoexp/my_exp.py \
    --work-dir experiments/train/autoexp/my_exp \
    --device auto > train.log 2>&1
```

チェックポイント取得：

```bash
ls experiments/train/autoexp/my_exp/best_coco_AP_epoch_*.pth | tail -1
```

## 実験結果の見方

`autoexp/results.tsv` はタブ区切りで git 追跡なし：

```
commit	bp_oks75	bp_oks50	mydata_oks75	status	description
a1b2c3d	0.801000	0.892000	0.612000	keep	baseline (curriculum_s2)
b2c3d4e	0.805000	0.893000	0.618000	keep	v1_weight=0.5 in VisibilityAwareSimCC
c3d4e5f	0.798000	0.890000	0.600000	discard	gau hidden_dims=128 (worse)
```

## 新しい実験 config を手動で書く場合

### 最小の例（config 変更のみ）

```python
"""GAU hidden_dims を 128 に削減して relation prior への依存を減らす。"""
_base_ = ["../../base/base_bizarre_pose.py"]

load_from = "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"
data_root = "data/merged_500_corrected/"

codec = dict(
    type="SimCCLabel",
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

model = dict(
    type="TopdownPoseEstimator",
    # ... (backbone, data_preprocessor 等は base から継承)
    head=dict(
        type="RTMCCHead",
        in_channels=768,
        out_channels=17,
        input_size=(192, 256),
        in_featuremap_size=(6, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=128,  # ← 変更点
            s=128, expansion_factor=2,
            dropout_rate=0.0, drop_path=0.0,
            act_fn="SiLU", use_rel_bias=False, pos_enc=False,
        ),
        loss=dict(type="KLDiscretLoss", use_target_weight=True, beta=10.0, label_softmax=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True),
)
```

### カスタム Python モジュールを使う場合

**必須**: config に `custom_imports` を追加する。なければ `KeyError: 'MyClass is not in the registry'` になる。

```python
"""VisibilityAwareSimCC で v1_weight=0.5 を使う例。"""
_base_ = ["../../base/base_bizarre_pose.py"]

custom_imports = dict(
    imports=["pose_estimation.codecs.visibility_aware_simcc"],
    allow_failed_imports=False,
)

load_from = "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"
data_root = "data/merged_500_corrected/"

codec = dict(
    type="VisibilityAwareSimCC",  # カスタム codec
    v1_weight=0.5,                # occluded keypoint weight
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)
# ... (model 等は必要に応じて)
```

既存のカスタムクラスと import パス：

| クラス | import パス |
|---|---|
| `OccludedRTMCCHead` | `pose_estimation.models.heads.occluded_rtmcc_head` |
| `VisibilityAwareSimCC` | `pose_estimation.codecs.visibility_aware_simcc` |
| `VisibilityWeightControl` | `pose_estimation.transforms.visibility_weight_control` |
| `RandomEdgesBlackout` | `pose_estimation.transforms.random_edges_blackout` |

## ベースライン

| Model | OKS@50 | OKS@75 |
|-------|--------|--------|
| 論文 best (Chen & Zwicker, WACV 2022) | 0.898 | 0.793 |
| **Curriculum S2 (現在の best)** | **0.892** | **0.801** |

目標: OKS@75 > 0.801 かつ mydata sanity 通過（OKS@75 ≥ 0.40）
