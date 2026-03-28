**もう主戦場は backbone ではない。**
あなたの表を見る限り、改善の主因は一貫して **表現学習より教師信号の整備** です。
RTMPose-m COCO → RTMPose-l HumanArt で大きく伸び、さらに Bizarre Pose の手修正 visibility と curriculum で伸びている。一方で backbone 総当たり、DINOv2、Amateur Drawings 追加は死んでいる。つまり今の系では、**「もっと強い特徴抽出器を探す」より、「何を正解としてどう学ばせるか」を詰めた方が効く**。RTMPose 自体も、CSPNeXt + SimCC + GAU という軽量設計で、局所座標と関節関係をどう学習させるかに寄ったモデルです。ここは元論文の設計思想とも整合しています。 

次に、あなたの4仮定は、現状こう順位づけできます。

**1位: 仮定1（visible / occluded を同列に扱う）**
これは、もう「候補」ではなく**主犯**です。
理由は単純で、元データの `v=2` 汚染を直し、occlusion curriculum を入れた時点でベストが出ているからです。しかも WACV 2024 の方向性とも一致している。あの論文も、occluded keypoint を visible と同列に扱うと visible 側まで壊しうることを示していて、occlusion reasoning 自体は有効だが、**雑な visibility 処理は逆効果**という立場です。あなたの「VisNet false negative で hip/ankle が悪化」はまさにそこです。
要するに、**遮蔽推論の着眼は正しいが、masking の挿し方が粗い**。そのせいで idea ではなく implementation が負けている。 ([CVFオープンアクセス][1])

**2位: 仮定4（自然画像由来の関節関係がそのまま使える）**
これもかなり強い。
COCO pretrained から HumanArt pretrained で大きく跳ね、さらに BP fine-tune で伸びている時点で、**関節関係 prior のズレ**は実在しています。特に 2–3頭身、逆さ、密集線画、極端な短足長腕でまだ破綻しているなら、問題は texture ではなく **人体比率・関節長・左右関係・可視性ルールの prior** です。
RTMPose の GAU は keypoint dependency を学ぶ設計ですが、その dependency が COCO / HumanArt 寄りに固まっているなら、anime 的な骨格比率に対してまだ硬い、という読みは自然です。これは backbone の不足ではなく、**relation prior のミスマッチ**です。 

**3位: 仮定2（crop 内に真値がある）**
これは「全体の主犯」ではないが、**外れ値の主犯**です。
あなたの OKS パターンがそれを示しています。
論文 best に対して **OKS@75 は上回るのに、OKS@50 は -0.006**。これは、平均的な局所化精度は改善している一方で、**まだ少数の失敗例を取りこぼしている**ことを意味します。普通、この形は「全体が少しずつ悪い」ではなく、**一部ケースで崩壊している**ときに出やすい。
バストアップで膝下が消える、画面外なのに座標を強制出力する、という挙動はまさにそのタイプです。ProbPose が presence / visibility / uncertainty を分けているのも、既存の top-down pose が **“存在しない / 画面外 / 不可視” を雑に一括処理している**からです。あなたの結果では、これはメイン平均精度より、**失敗 tail を削るための論点**に見えます。 ([CVFオープンアクセス][2])

**4位: 仮定3（各関節が単峰的に局所化できる）**
現時点では、これは**未立証**です。
beta=10→5 の失敗は、SimCC の単峰性が悪いことの証明ではありません。単に calibration を崩しただけの可能性が高い。
この仮定が本当に主犯なら、失敗は「少しずれる」ではなく、**左右反転、前後腕の取り違え、遮蔽部位の複数候補化、self-occlusion で複数峰がありそうなのに強引に1点へ潰れる**という形で出るはずです。今あなたが書いている事実だけでは、そこまではまだ見えていません。だからここに賭けるのは、今の段階ではまだ早いです。

要するに、今の証拠が示しているのはこうです。

**第一ボトルネックは supervision semantics。**
visibility 汚染、occluded の扱い、難度順学習。ここが一番効いている。

**第二ボトルネックは relation prior。**
自然画像で学んだ関節関係が、デフォルメ体型・極端姿勢でまだ壊れている。

**第三ボトルネックは tail failure。**
画面外・存在しない関節を「とりあえず出す」ことで少数の致命傷が残っている。

**そして backbone は少なくとも現時点では主犯ではない。**
14M の Curriculum S2 が 86.8M の論文 best を OKS@75 で超えている時点で、容量不足説はかなり弱い。
この結果はきれいです。能力が足りないのではなく、**どこを当てにいくべきかの定義が以前より正しくなった**だけです。

あと、**mydata を「汎化確認の独自ベンチマーク」と呼ぶのは条件付きで正しい**です。
正しいのは、少なくとも次の条件を満たす場合だけです。

* 学習データと画像・構図・派生元が重なっていない
* 実験中に mydata を見てハイパラ選択していない
* 失敗例を見てアノテーションや手法を修正するサイクルに使っていない
* 評価セットとして一度固定した後、後から足したり引いたりしていない

これを破っているなら、それはベンチマークではなく**開発用の私有検証セット**です。
悪いことではないが、意味は違う。研究として書くなら、この区別はごまかせません。

最後に、一番重要な解釈だけ言う。
あなたの結果は「RTMPose をもっと大きくすれば勝てる」ではなく、**RTMPose の仮定のうち、イラストで壊れている部分をすでにかなり特定できている**ことを示しています。
しかもその壊れ方は、かなり限定的です。全面崩壊ではない。
だから今の段階で architecture zoo をさまようのは鈍い。
数字を見る限り、問題はもう**モデル選び**ではなく、**visibility・relation・presence のどれが remaining error を支配しているか**の切り分けです。

[1]: https://openaccess.thecvf.com/content/WACV2024/papers/Sun_Rethinking_Visibility_in_Human_Pose_Estimation_Occluded_Pose_Reasoning_via_WACV_2024_paper.pdf "Rethinking Visibility in Human Pose Estimation: Occluded Pose Reasoning via Transformers"
[2]: https://openaccess.thecvf.com/content/CVPR2025/papers/Purkrabek_ProbPose_A_Probabilistic_Approach_to_2D_Human_Pose_Estimation_CVPR_2025_paper.pdf "ProbPose: A Probabilistic Approach to 2D Human Pose Estimation"
