**今の問題設定なら、RTMPose系を捨てて別系統へ飛ぶより、RTMPoseの表現と学習信号の噛み合わせを詰める方が筋が良い。** 根拠は単純で、HumanArt 事前学習の効果が、同系統内のモデルスケール差よりずっと大きいからです。Human-Art validation の GT bbox 条件では、RTMPose-l は COCO 事前学習で **56.4 AP**、HumanArt 事前学習版で **75.3 AP** まで伸びます。一方、HumanArt 事前学習済みどうしの **m→l** の差は **72.8→75.3 AP** 程度です。逆に COCO val では HumanArt 事前学習版が少し下がっており、**一般性能よりドメイン特化が効いている**と読めます。つまり、この領域ではまず「モデルを大きくするか」ではなく、「どのドメイン信号をどう学ばせるか」が主レバーです。 ([GitHub][1])

RTMPose 自体の性質も、その見方を支持します。RTMPose は **top-down** で人領域を切り出し、**CSPNeXt** バックボーンの上に、**SimCC の座標分類**、**7×7 conv**、**FC**、**GAU による keypoint 関係モデリング**を載せた構成です。学習面では **UDP 事前学習**、**strong→weak の2段 augmentation**、**temperature 付き soft label**、**separate σ** などで詰めています。つまりこれは元から「巨大表現を後で雑に使う」モデルではなく、**軽量な表現に対して、局所座標と関節関係をどううまく載せるか**で勝っているモデルです。したがって、イラスト特化でさらに詰めるなら、バックボーン総入れ替えより、**visibility・crop・関節関係・座標表現**のどこが現状の誤差源かを潰す方が、構造的に自然です。 

あなたの今の「occluded を少なくした集合で先に固めてから、難しいものへ移る」方針は、勘ではなくかなり合理的です。WACV 2024 の visibility 論文は、**occluded keypoints を visible と同列に混ぜて学習すると、visible keypoints の精度まで落ちる**と報告しており、特に小さい学習集合ではその悪影響が強く出ています。実際、COCO minitrain では visible-only 学習の方が **PCK-visible 83.4 vs 81.9**、**PCK-all 79.0 vs 77.7** でした。しかもその論文は、occlusion 処理の自然な挿し込み先が **keypoint token 間の attention** だと主張しています。RTMPose はちょうどそこに GAU を持っているので、あなたの現状ベストが「見える点を先に固める」側に寄っているのは、かなり筋が通っています。 ([CVFオープンアクセス][2])

さらに、bizarre-pose-estimator 側の結果を見ると、**追加データの効き方が appendage に偏っている**のも重要です。拡張 ADD は **4000枚**で、そのうち学習が **3200枚**ですが、追加した多様なポーズは特に **elbow / wrist / knee / ankle** の改善を押し上げています。要するに、このドメインの本当のボトルネックは「絵柄っぽさ」そのものより、**末端関節・遮蔽・極端ポーズ・関節間整合性**です。だから、今後の軸を「よりイラストっぽい backbone を探すこと」に置くのはややズレています。人間で言えば、服を替えても骨格は治らない、という程度の話です。 

この前提で、RTMPose 系で筋が良い改善軸は4つに整理できます。
**第1に、visibility を学習の副属性ではなく、構造上の制御信号として扱う軸。** これは今のあなたの流れと最も整合します。RTMPose は keypoint token を GAU で混ぜるので、visible と occluded を無差別に通すと、そもそも関節間伝播が汚れやすい。したがって改善の本命は、backbone 差し替えより **visibility-aware relation modeling** 側です。これは RTMPose の構造と正面から噛み合っています。 

**第2に、top-down crop 由来の境界問題を正面から扱う軸。** RTMPose は crop された activation window 内で座標分類をします。ところがイラストは、四肢の切れ、画面端への逃げ、極端なトリミング、デフォルメでの消失が多い。ProbPose は、既存の top-down HPE が **out-of-image keypoints を事実上まともに扱っていない**ことを問題化し、crop 系 augmentation や presence / visibility の明示が、**out-of-image だけでなく in-image の局所化も改善する**と示しています。あなたのデータで誤差が「境界近くの手首・足首・見切れ」に寄っているなら、これはかなり濃いボトルネックです。 ([CVFオープンアクセス][3])

**第3に、SimCC の“単一座標に落とす”表現そのものの限界を疑う軸。** RTMPose は Gaussian soft label、temperature、separate σ でかなり丁寧に調整されていますが、それでも本質は **1本の x と y を当てる座標分類**です。イラストでは、隠れている関節、左右の取り違え、前腕と上腕の圧縮、異常短足/長腕などで、真の不確実性が「少しぼやけた一点」では済まないことがあります。そこが支配誤差なら、問題は backbone ではなく **出力表現が曖昧さを持てないこと**です。ProbPose が probability / visibility / presence を分けているのも、この問題意識です。 

**第4に、HumanArt 以降の残差は“art domain gap”ではなく“anime pose semantics gap”だと見る軸。** Human-Art は **5 natural + 15 artificial scenarios、5万枚、12.3万 person instances** を持つかなり大きい橋渡し用データです。それでも、あなたがさらに bizarre 側のアノテーション修正と occlusion 制御で伸びているなら、残っているギャップは単なる「人工画像をもっと見せれば埋まる差」ではありません。より狭く言えば、**アニメ体型・省略表現・関節の可視/不可視ルール・左右対称破れ**の差です。ここを texture adaptation と誤認すると、延々と回り道します。 ([CVFオープンアクセス][4])

逆に、あまり筋が良くない方向もはっきりしています。
**巨大生成モデル系へ逃げること**、**backbone を片っ端から総当たりすること**、**visibility を単なるラベル追加で済ませること**です。SDPose 型は「COCO-only でも OOD に耐える」という別の勝負をしていて、あなたの目的である「イラストドメイン専用モデルを詰める」とは問題設定がずれています。今の条件では、改善余地は backbone の外よりむしろ **RTMPose の head / relation / supervision semantics** にあります。ここを無視して architecture zoo 巡りを始めると、研究ではなく願掛けになります。 ([GitHub][1])

要するに、あなたの現在地は悪くない。
そして次に考えるべき問いは「**もっと強いモデルは何か**」ではなく、**RTMPose のどの仮定がイラストで壊れているか**です。候補はかなり絞れます。
**visible/occluded を同列に扱う仮定、crop 内に真値がある仮定、各関節が単峰的に局所化できる仮定、自然画像由来の関節関係がそのまま使える仮定。**
この4つのどれが主犯かを見れば、方針は自然に決まります。

---

## 実験結果による検証 (2026-03-25 時点)

### ベンチマーク (Bizarre Pose test 487枚, 論文互換OKS)

| Model | OKS@50 | OKS@75 | params | 備考 |
|-------|--------|--------|--------|------|
| 論文 best (Feature Concat +new) | 0.898 | 0.793 | 86.8M | 2 backbone 並列 |
| 論文 Feature Matching +new | 0.895 | 0.791 | 9.9M | domain backbone 蒸留 |
| SDPose (参考, 50枚途中) | 0.899 | 0.794 | 950M | SD UNet, fine-tune なし |
| RTMPose-m COCO (pretrained) | 0.696 | 0.547 | 14M | fine-tune なし |
| RTMPose-l HumanArt (pretrained) | 0.860 | 0.744 | 28M | fine-tune なし |
| Stage A (COCO→BP) | 0.880 | 0.779 | 14M | |
| HumanArt→BP | 0.885 | 0.791 | 14M | |
| **Curriculum S2 (ours best)** | **0.892** | **0.801** | **14M** | |

OKS@75 で論文 best を上回り (0.801 vs 0.793)、OKS@50 は -0.006 で未到達。

### 4つの仮定の検証状況

**仮定1: visible/occluded を同列に扱う** → **主犯の一つと確認**
- Bizarre Pose は全 54,400 keypoints が v=2 (visible)。遮蔽された関節も「見える」として最高重みで学習
- 3200枚の visibility を手動修正 (15.2% → v=1)。curriculum learning と組み合わせて効果確認
- Occluded Reasoning (WACV 2024 ベース) を実装。3層 GAU で腕 +3% 改善だが masking の副作用あり
- **masking が見えている部位を壊す問題が未解決**: VisNet の false negative → hip/ankle 悪化

**仮定2: crop 内に真値がある** → **部分的に問題**
- mydata のバストアップ構図で膝下が画面外。モデルは画面外の keypoint にも座標を出力する (抑制機構なし)
- ProbPose (CVPR 2025) の presence/visibility 分離が有効な可能性あり → 未実装

**仮定3: 各関節が単峰的に局所化できる** → **未検証**
- SimCC の beta=10 が予測分布を極端に鋭くし、不確実な場合でも高 confidence で 1 点に集中
- beta=5 に下げる実験は chibi 悪化で失敗。ただし問題は beta ではなく SimCC の構造的制約の可能性
- ProbPose の確率的出力表現が候補だが実装コスト高

**仮定4: 自然画像由来の関節関係がそのまま使える** → **部分的に問題**
- GAU の 17 トークン attention は COCO の関節関係を学習済み。イラストのデフォルメ体型 (2-3頭身) では関節間比率が破綻
- HumanArt pretrained でこの問題が緩和 (COCO pretrained 0.696 → HumanArt 0.831)
- Bizarre Pose fine-tune でさらに緩和 (0.831 → 0.892)
- ただし mydata の extreme なポーズ (逆さ、密集線画) ではまだ破綻

### 確認済みの非効果

| 手法 | 結果 | 理由 |
|------|------|------|
| Amateur Drawings 178K 追加 | 全体悪化 | 子供の落書きはドメインが違いすぎる |
| EMA, Dropout, Grad Accum 等 7 手法 | 全て改善なし | curriculum S2 で既に正則化十分 |
| DINOv2 特徴量 | パーツ分離不可 | 写真ドメイン学習はイラストに転移しない |
| SDPose fine-tune | 過剰設計 | 0.95B params, CPU推論に1枚40秒 |
| backbone 総当たり (HRNet, UniFormer, ViTPose) | Curriculum S2 以下 | データ量が不足でモデルサイズの恩恵なし |

### やらないこと

- 根拠なしに一般的な学習テクニックを試す
- 写真ドメインの pretrained モデル特徴量をそのまま使う
- 複数の変更を同時に適用する
- 独自の評価指標で判断する (論文互換 OKS を使う)
- 巨大生成モデル系 (SDPose) に逃げる
- backbone を片っ端から総当たりする

### 次のステップ (優先順)

1. **masking の副作用を解消する**: visibility を hard masking (×0.01) ではなく soft weight として GAU attention に反映する。具体的には attention score に visibility を乗算し、occluded keypoint からの attention を弱めるが完全に消さない
2. **ProbPose の presence 概念を導入する**: 画面外の keypoint に対して「存在しない」を予測する機構を追加。SimCC の出力に presence head (binary) を付与
3. **mydata の GT を拡充する**: 50-100枚を目標に、多様な遮蔽パターンを含むテストセットを構築

[1]: https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/humanart/rtmpose_humanart.md "mmpose/configs/body_2d_keypoint/rtmpose/humanart/rtmpose_humanart.md at main · open-mmlab/mmpose · GitHub"
[2]: https://openaccess.thecvf.com/content/WACV2024/papers/Sun_Rethinking_Visibility_in_Human_Pose_Estimation_Occluded_Pose_Reasoning_via_WACV_2024_paper.pdf "Rethinking Visibility in Human Pose Estimation: Occluded Pose Reasoning via Transformers"
[3]: https://openaccess.thecvf.com/content/CVPR2025/papers/Purkrabek_ProbPose_A_Probabilistic_Approach_to_2D_Human_Pose_Estimation_CVPR_2025_paper.pdf "ProbPose: A Probabilistic Approach to 2D Human Pose Estimation"
[4]: https://openaccess.thecvf.com/content/CVPR2023/papers/Ju_Human-Art_A_Versatile_Human-Centric_Dataset_Bridging_Natural_and_Artificial_Scenes_CVPR_2023_paper.pdf "Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes"
