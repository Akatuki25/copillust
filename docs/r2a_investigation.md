# R2a 調査結果と検証プロトコル: 3D→スケッチ生成合成 (2026-07-03)

`docs/redesign_proposal.md` §3a の具体化。**生成サンプルを見て go/no-go を判断する**
ための調査結果・技術スタック・検証手順・判定基準。

## 1. 先行事例の調査結果 — 同型パイプラインは実証済み

「幾何条件 → 拡散生成 → 姿勢推定の学習データ」という構成の成功例が複数ある:

| 事例 | 構成 | 結果 |
|------|------|------|
| [Zero-shot pose estimation for prosthetic users](https://arxiv.org/pdf/2312.07854) (PLOS Digital Health 2025) | フレームの edge map → ControlNet で「健常者風」画像を生成、**元のポーズを保存** | keypoint 誤差 37-76% 削減 |
| [Synthetic-Child](https://arxiv.org/pdf/2603.02598) (2026) | GT pose 注入 + **dual ControlNet** 条件付けで子供姿勢の合成データパイプライン | 学習→エッジ配備まで成立 |
| [ControlEvents](https://arxiv.org/pdf/2509.22864) (2025) | **未知の2D骨格から**イベント画像を生成し姿勢推定器を学習 | 生成データのみで学習成立 |

→ 「拡散モデルは条件の幾何を保ったままドメインを変換でき、その出力で姿勢推定器を
学習できる」ことは当該分野で再現されている。未検証なのは
**「スケッチ・ラフ画風への変換で線の統計が mydata 系に届くか」** のみ。
これが R2a の検証対象。

## 2. 技術スタック (調査に基づく選定)

### 生成側

- **Base checkpoint: NoobAI-XL または Illustrious-XL 系**
  - Danbooru 全量で学習済みのため [`sketch` / `traditional media` /
    `colored pencil (medium)` / `graphite (medium)` / `monochrome` 等の
    style タグをネイティブに解する](https://civitai.com/articles/25464/common-style-tags-recognized-by-illustrious-and-other-danbooru-based-models)
  - → **初手は LoRA 自作不要。プロンプトタグだけでスタイル検証を開始できる**
- **ControlNet: [MistoLine](https://github.com/TheMistoAI/MistoLine)** (SDXL用, 第一候補)
  - 「手描きスケッチ・各種前処理・**モデル生成のアウトライン**」いずれの線入力にも
    頑健と明記 — 3D レンダ輪郭線を入力する本件のユースケースに合致
  - NoobAI 専用の [noob-sdxl-controlnet-lineart_anime](https://huggingface.co/Eugeoter/noob-sdxl-controlnet-lineart_anime) を対抗馬として比較
- **スタイル強化 (第2段, タグで不足の場合のみ)**: 既製 LoRA
  ([Rough Anime Sketch](https://civitai.com/models/561969/rough-anime-or-styles-or-sketch),
  [Anime Sketch SDXL](https://civitai.com/models/202764/anime-sketch-style-sdxl-and-sd15),
  [HandDrawnAnime XL](https://civitai.com/models/153174/handdrawnanime-xl)) → それでも
  不足なら Danbooru sketch タグ画像で自作 LoRA (第3段)
- **アナログ化後処理** (色鉛筆・紙・撮影系): 紙テクスチャ合成、パース warp、
  照明勾配、机背景 — OpenCV で手続き実装 (生成とは独立に検証可能)

### 3D 側

- **Blender + [VRM add-on](https://extensions.blender.org/add-ons/vrm/)**: VRM import、
  Python API でカメラ・ライトのランダム化、Freestyle/Solidify で輪郭線パス、
  toon シェーダパス、depth パス出力
- **ポーズ**: [Mixamo retarget (Auto Scale + Re-Target)](https://elvneko.com/posts/vroid-blender-mixamo/)
  または Rokoko Studio Live addon。静止ポーズはモーションの中間フレームサンプリング
- **正解生成**: bone head/tail をカメラ行列で 2D 投影 → COCO17 マッピング
  (VRM humanoid ボーンは規格で名前固定なので機械的に対応付く)。
  v=0/1/2 は frustum 判定 + ray-cast 遮蔽判定で自動付与
- **素体**: VRoid Studio で自作パラメトリック生成 5-10体 (頭身スライダで 2〜8)。
  自作なのでライセンス問題なし

### 実行環境

- 生成: Colab T4 (fp16 SDXL+ControlNet ≒ 10-20s/枚 → 数百枚 = 2-4 GPU時間)。
  diffusers スクリプトで一括生成 (ComfyUI は手動探索用)
- Blender レンダ: ローカル macOS で可 (toon+輪郭線は軽量、Eevee/Workbench)

## 3. R2a 検証プロトコル

### Step 0: アセット準備 (〜2日)
- VRoid Studio 素体 3体 (頭身 7 / 5 / 2.5)
- ポーズ 30種 (Mixamo 10モーション × 中間フレーム、立ち/座り/寝転び/腕上げ を含む)
- カメラ 3構図 (全身 / バストアップ=見切れ / 俯瞰)

### Step 1: Blender バッチレンダ (〜2日)
- 3体 × 30ポーズ × 3構図 = 270 シーン × {輪郭線, toon, depth} パス
- 正解 JSON (COCO17 2D座標 + v flag) を同時出力
- **この時点の輪郭線レンダ自体も学習ベースライン用に保存** (手続きレンダの対照群)

### Step 2: 生成マトリクス (〜2日, Colab)
270 シーンに対し以下を掃引 (計 ~1600枚):

| 軸 | 水準 |
|----|------|
| スタイル指定 | ①タグのみ (`sketch, rough sketch, monochrome`) ②タグ+既製LoRA ③`colored pencil (medium), traditional media` (アナログ系) |
| denoise strength | 0.4 / 0.6 / 0.8 |
| ControlNet | MistoLine / noob-lineart_anime (代表条件のみ両方) |

### Step 3: 測定 (〜2日)

1. **ラベル忠実度**: S2/P3 で生成画像を再推定し、条件 GT との PCK@0.1 を
   strength 別に曲線化 (品質ゲート通過率カーブ)。既存の
   `scripts/eval/error_analysis.py` の距離計算を流用
2. **スタイル距離**: Danbooru sketch タグ実画像 200枚 (参照セット、mydata は不使用)
   との CLIP 特徴距離。+ 目視用の**比較シート** (条件別グリッド画像) を生成
3. **対照群**: Step 1 の生の輪郭線レンダ (= 手続きレンダ) を同じ指標で測る
   — 生成合成が対照より mydata 系分布に近づいていることが最低条件

### 判定 (ユーザーが比較シートを見て決める)

go の目安 (いずれもある strength 水準において):
- 目視: ラフの線質 (強弱・重ね線・未閉合) が出ており、手続きレンダとの差が明確
- ラベル忠実度: 品質ゲート通過率 ≥ 30% (生成は安価なので 3倍過剰生成で回る)
  かつ通過サンプルの目視ラベル正しさ ≥ 90% (20枚抜き取り)
- 頭身 2.5 の chibi 条件で四肢が人体比率に「戻されない」こと (Illustrious 系の
  prior が chibi をどこまで許すかが最大の不確実性)

no-go 時の分岐:
- 線質が出ない → LoRA 自作 (第3段) を1回だけ試す
- chibi が壊れる → chibi 専用 LoRA or 通常頭身のみで進め chibi は別道
- ラベル忠実度が全 strength で不足 → dual conditioning (lineart+depth,
  Synthetic-Child 方式) を追加して1回だけ再試行
- それでも不可 → 生成合成路線を放棄し、能動学習アノテーション
  (redesign_proposal §3a フォールバック) へ

## 4. 実施結果 (2026-07-04, ローカル M3 Pro で核心条件のみ先行検証)

パイプライン実装: `experiments/synth/` — blender_render.py (126シーンレンダ済み、
GT はオーバーレイ検証済み) / generate_sketch.py / measure_fidelity.py / make_sheet.py

生成4枚 (seed_san wave normal / walk chibi × cs 0.55 / 0.9) での所見:

1. **chibi の頭身は保持された** (最大の不確実性が解消)。頭身2.5相当の骨格条件でも
   拡散 prior は四肢を人体比率に戻さず、chibi らしい大きな目の顔で描き直した
2. **cs=0.55 で線が本物のラフに近づく**: 重ね線・引き直し・輪郭の乱れが出る。
   cs=0.9 は「丁寧な鉛筆画」寄り。トレードオフは設計通り存在
3. **ラベル忠実度は手続きレンダと同等**: S2 再推定 PCK@0.1 で line(対照) 0.88 /
   cs0.9 0.85 / cs0.55 0.85、全条件 PCK@0.2=1.0 (粗大ドリフトなし)。
   chibi は対照の line 自体が 0.76 → 差分は checker (S2) の chibi 弱さ由来
4. 局所的な再解釈はある (腰の小物→手袋、素足→靴)。関節位置には非影響だが
   規模生成時は品質ゲートで拾う前提を維持

技術メモ:
- **MPS では enable_attention_slicing() が step0 から latent を 100% NaN にする**
  (真っ黒画像)。fp16/bf16 無関係。無効化で解決。dtype は bf16 を使用
- SDXL fp16 VAE は fp16-fix VAE (madebyollin) に差し替え
- 168-214 s/枚 (768×1024, 28step, M3 Pro 18GB)。残りマトリクスは Colab T4 推奨
- 素体3体 (VRoid fem/masc 素体 + Seed-san 服付き)。fem/masc は服なし素体のため
  服の線密度検証は seed_san のみ → 規模化前に服付き VRM の追加が必要

## 5. 残る未確認事項 (実験前に潰す)

- [ ] NoobAI-XL / Illustrious のライセンス (fair-ai 系。学習データ生成への利用条件)
- [ ] MistoLine が「整いすぎた輪郭入力」からどこまで線を崩せるか
  (ControlNet weight を下げる余地も掃引軸に入れる)
- [ ] Danbooru sketch タグ参照セットの収集手段 (danbooru API / 既存 dump)
- [ ] Colab での VRAM 実測 (SDXL+ControlNet+LoRA fp16 で T4 16GB に収まる想定)
