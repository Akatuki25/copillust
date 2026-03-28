# RTMPose macOS ローカル実験ガイド

## 目的

このガイドは、次の2系統のイラストを対象に、**RTMPose を土台に fine-tune して学習が進むか**を macOS ローカル環境で検証するための手順をまとめたもの。

* **前者**: 完成イラスト。最低限「取れるべき」ライン
* **後者**: 線画・低頭身・partial body で、既存 SOTA でも失敗しやすいライン

主眼は次の3点。

1. 既存 RTMPose ベースラインの失敗を再現する
2. 独自ラベル付きデータの追加学習で改善傾向が出るか確認する
3. その結果が後段の誤り検知器に使える程度に安定するか確認する

---

ローカルでは、**Python 3.10 系 + conda + MMPose 1.x** で回すのが安全。

---

## 1. 推奨方針

### 採用モデル

* **第1段階**: `RTMPose-m`
* **第2段階**: うまく進むなら `RTMPose-l`

理由:

* `m` は軽く、学習が進むかの確認に向く
* `l` はより高精度だが、Mac では速度とメモリで厳しい場合がある
* まずは **GT bbox / loose crop 前提** で detector 問題を切り離す

### この段階で捨てるもの

* いきなり RTMO に行くこと
* いきなり detector-free 本番系に行くこと
* dense face / full segmentation / critique ラベルを最初から全部入れること

---

## 2. macOS 動作要件

### ソフトウェア

* macOS 12.3+
* uvで環境作成
* Miniconda または Mambaforge
* Python 3.10

### 備考

PyTorch は macOS で MPS を使えるが、OpenMMLab 側は Linux/CUDA 中心。
MPS で不安定なら CPU fallback で最初の smoke test を通すのが安全。

---

## 3. 必要なダウンロード一覧

### リポジトリ

1. `open-mmlab/mmpose`
2. 必要なら `open-mmlab/mmdetection`
3. 比較用として `ShuhongChen/bizarre-pose-estimator`
4. `IDEA-Research/HumanArt`

### 学習ベース

* RTMPose の body 17-kpt pretrained checkpoint（COCO 系）
* 可能なら Human-Art で学習済み RTMPose checkpoint

### データ

1. **Human-Art**
   art/artificial domain への適応元

2. **AnimeDrawings / bizarre-pose-estimator 側の公開データ**
   illustrated character pose の初期土台

3. **自前データ**
   今回のラベル仕様で追加したイラスト群
   chibi / lineart / partial-body / clutter を意図的に厚く入れる

---

## 4. ディレクトリ構成

```text
~/work/illust-pose/
  mmpose/
  data/
    HumanArt/
    bizarre_pose/
    custom_illust/
      images/
        train/
        val/
        test/
      annotations/
        train.json
        val.json
        test.json
  experiments/
    rtmpose_m_illust_v1/
```

---

## 5. conda 環境構築

```bash
conda create -n illust-pose python=3.10 -y
conda activate illust-pose

# Apple Silicon で MPS を使う場合
pip install torch torchvision torchaudio

# OpenMMLab 基本
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1,<2.2.0"
mim install "mmdet>=3.1.0,<3.4.0"

# MMPose
pip install -v -e ./mmpose

# 追加ツール
pip install pycocotools opencv-python matplotlib pandas jupyter scikit-image tqdm
```

### 動作確認

```bash
python -c "import torch; print(torch.__version__); print('mps', torch.backends.mps.is_available())"
python -c "import mmcv, mmengine, mmdet, mmpose; print(mmcv.__version__)"
```

---

## 6. データセット仕様

### 全件必須

* bbox
* COCO17 body keypoints
* 各 keypoint の state

  * `visible`
  * `occluded`
  * `out_of_frame`
  * `ambiguous`
  * `not_applicable`
* head 4 点

  * `head_top`
  * `chin`
  * `left_head`
  * `right_head`
* `render_type`
* `body_type`
* `frame_type`
* `background_complexity`
* `view_type`

### hard subset のみ

* coarse silhouette mask
* dense face landmarks（必要な subset のみ）

### 人手で打たないもの

* 肩幅異常
* 頭身異常
* 四肢長異常
* 関節角異常

これらは keypoints と head 4 点から後で計算する。

---

## 7. COCO 互換 JSON 例

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "train/sample_0001.png",
      "width": 768,
      "height": 1024,
      "render_type": "lineart",
      "background_complexity": "plain",
      "view_type": "three_quarter"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [120, 90, 420, 820],
      "num_keypoints": 13,
      "keypoints": [
        350,160,2,
        336,148,2,
        370,149,2,
        320,150,1,
        388,151,1,
        300,260,2,
        400,258,2,
        280,360,2,
        430,340,2,
        250,430,2,
        455,405,2,
        320,470,2,
        402,468,2,
        310,0,0,
        395,0,0,
        0,0,0,
        0,0,0
      ],
      "kp_states": [
        "visible","visible","visible","occluded","occluded",
        "visible","visible","visible","visible","visible",
        "visible","visible","visible","out_of_frame","out_of_frame",
        "out_of_frame","out_of_frame"
      ],
      "head_points": {
        "head_top": [352, 110],
        "chin": [351, 225],
        "left_head": [300, 160],
        "right_head": [404, 162]
      },
      "body_type": "chibi",
      "frame_type": "truncated_bottom"
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "keypoints": [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
      ],
      "skeleton": [
        [6,8],[8,10],[7,9],[9,11],[6,7],
        [6,12],[7,13],[12,13],[12,14],[14,16],[13,15],[15,17]
      ]
    }
  ]
}
```

---

## 8. 学習前に baseline を取る

### 8-1. 公式 RTMPose の baseline inference

まず pretrained RTMPose でそのまま 2 枚を通す。
この時点では **GT bbox を手で与える**。

* 前者: 取れるなら「最低限の成立ライン」
* 後者: 落ちるなら想定通り。以後の改善対象

### 8-2. 2 枚だけで判断しない

この 2 枚はケーススタディ。
判定は challenge split 全体で行う。
最低限 test split を以下の 4 群に切る。

* chibi
* lineart/sketch
* partial-body
* cluttered-background

---

## 9. 学習の段階設計

### Stage A: smoke test

* base: RTMPose-m
* input: 256x192
* bbox: GT bbox / loose crop
* epoch: 5〜10
* data: 自前データのみでもよいが、枚数が少なければ Human-Art と混ぜる

目的:

* loss が下がるか
* lineart / chibi / partial で collapse しないか
* baseline より visible-joint PCK が改善するか

### Stage B: actual fine-tune

* base: Stage A の best
* data: Human-Art + bizarre pose + custom_illust
* sampling: custom_illust を oversample
* epoch: 20〜50

目的:

* challenge subset で相対改善を確認
* 前者画像を安定して取れるか
* 後者画像で upper-body / face 軸が少なくとも破綻しないか

### Stage C: 必要なら RTMPose-l

Stage B で改善が見えたら、同じ recipe を RTMPose-l に移植。

---

## 10. 推奨 train コマンド

```bash
cd ~/work/illust-pose/mmpose

python tools/train.py \
  projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
  --work-dir ../experiments/rtmpose_m_illust_v1 \
  --cfg-options \
    train_dataloader.dataset.data_root=../data/custom_illust \
    train_dataloader.dataset.ann_file=annotations/train.json \
    train_dataloader.dataset.data_prefix.img=images/train/ \
    val_dataloader.dataset.data_root=../data/custom_illust \
    val_dataloader.dataset.ann_file=annotations/val.json \
    val_dataloader.dataset.data_prefix.img=images/val/ \
    test_dataloader.dataset.data_root=../data/custom_illust \
    test_dataloader.dataset.ann_file=annotations/test.json \
    test_dataloader.dataset.data_prefix.img=images/test/
```

### 実務上の注意

このままだと metainfo や dataset_type の差し替えが必要になることが多い。
最初は config をコピーして、`configs/custom/rtmpose_m_illust_v1.py` を別に作る方が安全。

---

## 11. test / inference コマンド

### val/test 評価

```bash
python tools/test.py \
  configs/custom/rtmpose_m_illust_v1.py \
  ../experiments/rtmpose_m_illust_v1/best_coco_AP_epoch_*.pth \
  --work-dir ../experiments/rtmpose_m_illust_v1_eval
```

### 単画像推論

GT bbox 前提にしたいなら、事前に crop した画像を通すのが簡単。

```bash
python demo/inferencer_demo.py \
  ../samples/front_case.png \
  --pose2d human \
  --draw-bbox \
  --show-progress
```

独自学習済みモデルで見る場合は、config と checkpoint を直接指定する。

---

## 12. 見るべき数値

全体 AP だけ見ると判断を誤る。
最低限、以下を分けて見る。

1. overall AP / PCK
2. chibi subset の visible-joint PCK
3. lineart subset の visible-joint PCK
4. partial-body subset の visible-joint PCK
5. no-pred / collapse 率
6. 左右の取り違え率
7. out_of_frame を visible と誤判定した率

### go / no-go

次へ進む基準:

* train/val loss が素直に下がる
* chibi / lineart / partial-body で baseline より相対改善
* 前者画像で major joints が安定
* 後者画像で upper-body / face 軸が少なくとも一貫する

これを満たさないなら、後段の誤り検知へ進むのは早い。

---

## 13. 2 枚の画像に対する現実的な期待値

### 前者画像

* 完成イラスト
* 背景は軽い
* 全身は大部分見えているが、ぬいぐるみ・衣装・腕で一部遮蔽あり

期待:

* fine-tune 後なら、**最低限ここは取れるべき**
* ただし脚の正確な末端は遮蔽物のせいで不安定でも不思議ではない

### 後者画像

* 線画
* 低頭身寄り
* 下半身が見えない
* 背景に人物の一部らしき別要素あり

期待:

* full-body pose を完全に出すことは要求しない
* **upper-body の首・肩・肘・顔軸が一貫するか** を見る
* `frame_type=truncated_bottom` と `state=out_of_frame` の学習が効いているかが本質

この画像で「見えていない脚を正しく hallucinate できるか」を最初の成功条件に置くのは間違い。

---

## 14. 推奨する実験順

1. pretrained RTMPose-m baseline を 2 枚に通す
2. GT bbox / crop 付きで baseline 結果を保存
3. custom_illust の最小 split で smoke test
4. challenge subset ごとに PCK と collapse 率を見る
5. 改善が出たら Human-Art 混合で本格 fine-tune
6. その後に RTMPose-l を試す

順番を逆にすると、何が効いたか分からなくなる。

---

## 15. まず揃えるべき最小セット

### 必須

* MMPose 本体
* RTMPose-m config
* RTMPose-m pretrained checkpoint
* custom_illust train/val/test
* 前者/後者のケース画像

### 強く推奨

* Human-Art
* bizarre-pose-estimator の公開データ
* challenge subset ごとの集計スクリプト

---

## 16. 失敗しやすい点

* Python 3.12/3.13 にする
* MMPose / mmcv / mmdet の version を雑に混ぜる
* 自前データを COCO 互換にせず独自形式のまま突っ込む
* lineart / chibi / partial-body を train にほぼ入れない
* 全体 AP だけ見て満足する
* 後者画像で full-body を最初の合格条件にする

---

## 17. 今の判断

今回の 2 画像を基準にするなら、最初の成功条件はこうです。

* **前者**: baseline より明確に安定して body pose が出る
* **後者**: full-body ではなく、head + neck + shoulders + elbows の整合が改善する

この条件を満たせば、pose 側は「後段の誤り検知へ接続してよい」と判断できる。

---

# 実験チェックリスト

```text
[ ] macOS 12.3+ / Apple Silicon / Xcode CLT
[ ] conda env with Python 3.10
[ ] torch + torchvision installed, MPS availability checked
[ ] mmengine / mmcv / mmdet / mmpose installed
[ ] RTMPose-m config available
[ ] pretrained checkpoint downloaded
[ ] custom_illust COCO-format JSON prepared
[ ] GT bbox / loose crop generated for train/val/test
[ ] baseline inference on the two target images saved
[ ] smoke test run finished
[ ] subset metrics computed: chibi / lineart / partial / clutter
[ ] qualitative comparison sheet saved
```
