なので、**具体的に使う想定のデータ**を、役割ごとに切って明示する。
結論から言うと、**自前で全部は作らせない**。
使うのは **既存の公開データ 3 本 + 小さい自前セット 1 本** です。しかも、自前は「全部」ではなく、**公開データで埋まらない穴だけ** を埋めるために使う。Human-Art は 50,000 枚・20 シナリオの human-centric データで、COCO 互換 JSON を持ち、`training_humanart.json` や `training_humanart_coco.json`、`training_humanart_cartoon.json` のような分割が最初から用意されています。Bizarre Pose は anime/manga 系の 17 COCO keypoints + bbox を持つ 4,000 枚の illustrated character pose dataset で、公開 repo から `bizarre_pose_dataset.zip` を落として使う前提です。Amateur Drawings は 178,000〜180,000 件規模の drawings に bbox・segmentation mask・joint location annotations が付いています。 ([GitHub][1])

## 使う想定のデータはこれ

### 1. ベース重み

まず土台は **MMPose の RTMPose-m の既存 pretrained checkpoint**。
MMPose は RTMPose を公式にサポートしていて、custom dataset を COCO 互換で載せられる。ここは 0 から学習しない。 ([GitHub][2])

### 2. supervised の主データ

ここが本体です。

#### A. Human-Art

役割は **artificial / non-photo domain への広い適応**。
これを入れる理由は、RTMPose 系でも COCO だけより Human-Art を混ぜた方が Human-Art validation で大きく伸びているからです。公式の Human-Art README では、RTMPose-m は 0.532→0.728、RTMPose-l は 0.564→0.753 まで上がっています。さらに Human-Art は COCO 互換で、`training_humanart.json`、`training_humanart_coco.json`、`training_humanart_cartoon.json` などの JSON が用意されています。 ([GitHub][1])

**使い方**
最初の supervised fine-tune では

* `training_humanart_coco.json` を使って broad adaptation
* その後、必要なら `training_humanart_cartoon.json` を重めにサンプリング
  にします。
  理由は、いきなり anime だけに寄せると、線画・人工表現・partial-body に対する一般化が痩せるからです。これは私の設計判断です。

#### B. Bizarre Pose Dataset

役割は **anime / manga キャラ絵の主 supervised データ**。
ここはコアです。Bizarre Pose は AnimeDrawingsDataset を拡張した illustrated character pose dataset で、17 COCO keypoints と bbox を持つ 4,000 サンプルです。repo README でも、`bizarre_pose_dataset.zip` を `_data` に展開する形が明記されています。 ([Papers with Code][3])

ただし、ここにははっきりした限界がある。
repo の FAQ 自体が、**“multiple characters や full-body でない画像には対応していない。single full-body characters に focused”** と書いています。つまり、**前者のような完成イラストには近いが、後者のような partial-body 線画にはそのままでは足りない**。だからこれだけで済ませる計画は成立しません。 ([GitHub][4])

#### C. Amateur Drawings Dataset

役割は **線画・drawn figure・人体検出失敗耐性の補助**。
Amateur Drawings は Meta/Facebook Research の Animated Drawings 由来のデータで、約 178k〜180k 件の amateur drawings に bbox、segmentation mask、joint annotations が付いています。repo には annotations と images のダウンロード方法まで明記されています。 ([GitHub][5])

ただし、これもそのまま主 supervised データにする気はない。
理由は、**絵柄は line drawing に近いが、anime character 的な比率や装飾とは分布が違う**からです。
なので使い方は限定します。

* 第一優先ではなく **補助 supervised / auxiliary** に使う
* 特に **lineart subset** と **partial / upper-body subset** をフィルタして使う
* mask があるので、**coarse silhouette auxiliary loss** にも回せる

要するに、これは **後者画像に近い failure mode を補強するための public source** です。

## 3. 自前で用意するデータ

ここはゼロにはできません。
ただし、**全部ではない**。
本当に必要なのは **公開データで欠けている交差領域** だけです。

その交差領域はこれです。

* chibi / super-deformed
* lineart / rough sketch
* partial-body / bottom-truncated
* anime / VTuber 的デザイン
* あなたが最終的に使いたい作風分布

Bizarre Pose は single full-body 寄りで、Amateur Drawings は line drawing だが anime キャラ分布ではない。
だから、**後者画像のような条件**をちゃんと評価したいなら、そこだけは自前で埋める必要がある。これは公開データの穴であって、私の趣味ではない。 ([GitHub][4])

### 自前の量

ここは最初から大きくしない。
**最初の run に必要なのは 200〜400 枚程度**で十分です。
推奨はこうです。

* train: 240
* val: 60
* test: 100

この 400 枚のうち、半分以上を
**chibi + lineart + partial-body** に寄せる。
前者・後者の2枚は当然 test の固定ケースに入れる。

### 自前データのソース

ここは具体化する。
画像の出所は、**あなたの手持ちイラスト群**があるならそれが最優先。
無い場合は、Bizarre Pose repo でも Danbooru 由来のデータを強く使っており、foreground や filtering も Danbooru ベースです。したがって、**Danbooru からタグ条件で絞った収集**が最も自然です。repo 側も Danbooru dataset を参照するよう明記しています。 ([GitHub][4])

自前 400 枚の収集条件はこう切ります。

* `chibi` / `super_deformed`
* `lineart` / `sketch`
* `upper_body` / `cowboy_shot` / `portrait`
* `solo`
* `simple_background` と `detailed_background` を両方
* 可能なら `sitting` / `foreshortening` / `crossed_legs` など崩れやすい構図

ここは **人手で全部描かせる話ではない**。
既存画像を集めて、今回のラベルだけ打つ話です。

## 4. 何を train に使い、何を test に使うか

ここも固定する。

### train

* Human-Art: `training_humanart_coco.json`
* Bizarre Pose: train split
* Amateur Drawings: lineart/upper-body/filter 後 subset
* Custom challenge train

### val

* Human-Art: `validation_humanart.json`
* Bizarre Pose: val split
* Custom challenge val

### test

* Human-Art: `validation_humanart_cartoon.json` を **public cartoon test** として使う
* Bizarre Pose: test split を **public anime full-body test** として使う
* Custom challenge test を **本命 test** にする
* その中に **あなたの前者・後者画像** を固定ケースとして含める

この切り方なら、

* broad art domain
* anime full-body
* lineart / partial-body / chibi
  の3系統を分けて見られる。Human-Art 側が scenario-specific JSON を持っているのは、この切り方に都合がいい。 ([GitHub][1])

## 5. 学習に使うラベルの現実的な扱い

ここも机上で済ませない。

### 17 body keypoints

これは全データで主タスク。
Bizarre Pose は最初から COCO17。Human-Art は 21 点なので、**COCO17 へ落とす converter** を書く。
MMPose は custom dataset / keypoint converter 前提なので、これは標準ルートです。 ([GitHub][2])

### head 4 点

これは **public 全部に要求しない**。
ここが重要。
前に提案したラベル仕様をそのまま全 public dataset に強制すると破綻する。
だから head 4 点は **custom challenge set にだけ入れる**。
loss は masked にして、head4 があるサンプルだけ auxiliary head を学習する。
これなら実行可能。

### silhouette mask

これも **Amateur Drawings と custom のみ**。
Human-Art と Bizarre Pose には要求しない。
これも auxiliary loss 扱い。

つまり、
**public データは各々が持っている範囲だけ使い、欠けるラベルは masked loss で無視する**。
これが現実的な実装です。

## 6. 最初の実験 recipe まで具体化するとこうなる

### Phase 1

RTMPose-m + COCO pretrained を出発点に、
**Human-Art + Bizarre Pose** だけで 17kp fine-tune。

目的は、

* art / anime 分布で普通に loss が下がるか
* 前者画像で body pose が安定するか

### Phase 2

Phase 1 の best から、
**Human-Art + Bizarre Pose + filtered Amateur Drawings + custom set** で継続学習。
ここで初めて

* lineart robustness
* partial-body robustness
* head4 auxiliary
* silhouette auxiliary
  を入れる。

### データ配分の推奨

これは私の提案だが、最初の run はこうする。

* Human-Art: 40%
* Bizarre Pose: 30%
* Amateur Drawings filtered: 15%
* Custom challenge: 15%

理由は、custom を増やしすぎると過学習、減らしすぎると後者画像の条件に効かないからです。

## 7. 端的な答え

**具体的に使う想定**はこれです。

1. **ベースモデル**
   MMPose の **RTMPose-m COCO pretrained**。 ([GitHub][2])

2. **主 supervised データ**
   **Human-Art**。具体的には `training_humanart_coco.json` と、評価用に `validation_humanart.json` / `validation_humanart_cartoon.json`。 ([GitHub][1])

3. **anime 専用 supervised データ**
   **Bizarre Pose Dataset**。具体的には repo が案内している `bizarre_pose_dataset.zip`。ただし full-body 中心なので、後者のような画像はここだけでは埋まらない。 ([GitHub][4])

4. **lineart / drawing 補助データ**
   **Amateur Drawings Dataset**。具体的には `amateur_drawings_annotations.json` と `amateur_drawings.tar`。主に lineart / upper-body / mask 補助用。 ([GitHub][5])

5. **自前データ**
   **200〜400 枚だけ**。
   公開データで欠けている **chibi + lineart + partial-body + anime** の交差領域だけ埋める。
   ここにあなたの2枚も固定テストとして入れる。

これが、今の目的に対して **実行可能な最小構成** です。
前のように「自前で全部作る」話ではないし、逆に「公開データだけで何とかなる」とも言っていない。
両方の中間で、実際に回るラインです。

[1]: https://github.com/IDEA-Research/HumanArt?utm_source=chatgpt.com "GitHub - IDEA-Research/HumanArt: [CVPR 2023] The official implementation of CVPR 2023 paper \"Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes\""
[2]: https://github.com/open-mmlab/mmpose?utm_source=chatgpt.com "GitHub - open-mmlab/mmpose: OpenMMLab Pose Estimation Toolbox and Benchmark."
[3]: https://paperswithcode.com/dataset/bizarre-pose-dataset?utm_source=chatgpt.com "Bizarre Pose Dataset Dataset | Papers With Code"
[4]: https://github.com/ShuhongChen/bizarre-pose-estimator "GitHub - ShuhongChen/bizarre-pose-estimator: WACV2022: Transfer Learning for Pose Estimation of Illustrated Characters · GitHub"
[5]: https://github.com/facebookresearch/AnimatedDrawings "GitHub - facebookresearch/AnimatedDrawings: Code to accompany \"A Method for Animating Children's Drawings of the Human Figure\" · GitHub"
