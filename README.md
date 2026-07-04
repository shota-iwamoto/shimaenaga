# Shimaenaga

**Attentive Histogram GBDT — サンプル単位のトークン注意機構を備えた勾配ブースティング**

Shimaenaga は、LightGBM スタイルのヒストグラムベース GBDT に、サンプルごとの
トークン注意機構(attention)を組み込んだ機械学習ライブラリです。特徴量を
「トークン」と呼ぶサブセットに分割して各トークンごとに木を成長させ、
サンプルごとに学習された注意重みでそれらを混合します。これにより、
純粋な GBDT では捉えにくい特徴グループ間の相互作用を表現できます。

- コアは **C++17**(依存ライブラリなし、OpenMP 並列)
- Python から **scikit-learn 互換 API** で利用可能(`fit` / `predict` /
  `predict_proba` / `clone` / `GridSearchCV` / `cross_val_score` 対応)
- 回帰(L2 / Huber / Quantile / MAE)・二値分類・多クラス分類・
  ランキング(LambdaMART)をサポート

---

## アーキテクチャ概要

学習は 2 相構成です。Phase A で木構造を成長させ、Phase B で
Newton 法によるパラメータの再適合(refit)を行います。

```
入力 X  ──→  [BinMapper]  ──→  ビン化された特徴

Phase A ─── 各トークン p ごとにツリー成長 (LightGBM leaf-wise)
             ゲートツリー(ルーティング用)も同時に成長

Phase B ─── Newton refit(反復最適化)
             B1: 葉値 v_{pℓk}
             B2: 読み出し QK 埋め込み (q, k, b)
             B3: 自己注意 QK(Tier-2 のみ)
             B4: ヘッド重み ρ_h
```

### 注意機構

```
トークン p: 特徴のサブセット S_p から学習した「部品」
ゲート:     α_{ihp} = softmax(q_{h,gate_leaf}^T k_{h,p,token_leaf} / √d_a)
混合:       β_{ip}  = Σ_h ρ_h · α_{ihp}   (サンプルごとの部品重み)
出力:       φ_{ik}  = Σ_p β_{ip} · v_{p,leaf_p(i),k}
```

各サンプルがどのトークン(特徴グループ)をどれだけ重視するかを、
ゲートツリーと QK 埋め込みが決定します。

### Tier システム

| Tier | 機能 | 用途 |
|------|------|------|
| 0 | 純粋 GBDT(LightGBM 退化形) | ベースライン、最速 |
| 1 | + Attentive Readout | 特徴グループ間の相互作用 |
| 2 | + Self-Attention(Tier-1 の上) | 複雑なトークン間依存関係 |

tier-1/2 の表現力は tier-0 を常に包含するよう設計されています
(最初のトークンは全特徴を使用)。

---

## ビルドとインストール

### 必要環境

- C++17 コンパイラ(GCC ≥ 9, Clang ≥ 11, MSVC 2019 以降)
- CMake ≥ 3.20
- Python ≥ 3.8 + NumPy
- scikit-learn(sklearn 互換 API と examples の実行に使用)
- (オプション)pybind11 — ネイティブ拡張としてビルドする場合のみ

### 1. C++ コアのビルド(必須)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

これで共有ライブラリ `build/libshimaenaga_core.dylib`(macOS)/
`.so`(Linux)が生成されます。

### 2. Python パッケージの利用

Python パッケージ `shimaenaga` は **pybind11 なしでも動作します**。
pybind11 拡張が見つからない場合、ビルド済み共有ライブラリを ctypes 経由で
自動的に呼び出します(機能は同一)。

```bash
# 方法 A: そのまま使う(ビルド済み共有ライブラリを自動検出)
PYTHONPATH=python python3 -c "import shimaenaga; print(shimaenaga.__version__)"

# 方法 B: 開発インストール
pip install -e .

# 方法 C: pybind11 ネイティブ拡張としてビルド
pip install pybind11
pip install -e .
```
---

## クイックスタート

```python
from shimaenaga import ShimaenagaRegressor, ShimaenagaClassifier, ShimaenagaRanker
```

### 回帰

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from shimaenaga import ShimaenagaRegressor

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = ShimaenagaRegressor(
    tier=1,             # Tier-1: Attentive Readout
    num_tokens=4,       # 特徴を 4 グループに分割
    num_heads=2,        # 注意ヘッド数
    d_attn=4,           # 注意埋め込み次元
    num_iterations=300,
    learning_rate=0.05,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 二値分類

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from shimaenaga import ShimaenagaClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

clf = ShimaenagaClassifier(
    num_class=1,        # 1 = 二値(既定)
    tier=1,
    num_tokens=4,
    num_iterations=200,
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)   # shape (n, 2)
labels = clf.predict(X_test)
```

### 多クラス分類

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from shimaenaga import ShimaenagaClassifier

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

clf = ShimaenagaClassifier(
    num_class=3,        # K クラス softmax
    tier=1,
    num_tokens=2,
    num_iterations=100,
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)   # shape (n, 3)
```

### ランキング(LambdaMART)

```python
from shimaenaga import ShimaenagaRanker

# group: クエリごとのサンプル数(合計 = len(X_train))
model = ShimaenagaRanker(tier=1, num_iterations=200)
model.fit(X_train, y_train, group=group_train)
scores = model.predict(X_test)

# 検証セット付き(eval_group 必須 — valid の NDCG をクエリ単位で計算するため)
model.fit(X_train, y_train, group=group_train,
          eval_set=[(X_valid, y_valid)], eval_group=[group_valid],
          early_stopping_rounds=50)
```

### 注意重みの取得(解釈性)

```python
diag = model.attention_diagnostics(X_test)
beta = diag["beta"]      # shape (n, num_tokens): 各サンプルのトークン重み
                         # (ブースティングブロック平均、行和 = 1)
```

### Early Stopping

```python
model = ShimaenagaRegressor(num_iterations=1000, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
print(model.best_iteration_)
```

### モデルの保存・ロード

```python
model.save_model("model.sbb")       # バイナリ形式 (.sbb, 形式 v3)

# ロード: 同じ設定でインスタンスを作り、一度 fit で booster を初期化してから load
loaded = ShimaenagaRegressor(tier=1, num_tokens=4)
loaded.fit(X_tiny, y_tiny)          # booster の初期化(小さなデータで可)
loaded.load_model("model.sbb")
pred = loaded.predict(X_test)
```

---

## パラメータ一覧

既定値はコンストラクタ(`python/shimaenaga/sklearn_api.py`)および
C++ `Config` と一致しています。

### 基本設定

| パラメータ | 既定値 | 説明 |
|-----------|-------|------|
| `objective` | `"regression"` | `"regression"` / `"huber"` / `"quantile"` / `"mae"` / `"binary"` / `"multiclass"` / `"lambdarank"` |
| `num_iterations` | 1000 | ブースティング反復数 |
| `learning_rate` | 0.05 | 学習率 η |
| `num_class` | 1 | クラス数(Classifier のみ。1=二値, K>1=多クラス) |
| `huber_alpha` | 0.9 | Huber の適応的 δ = 残差絶対値の α 分位点(毎反復更新) |
| `quantile_alpha` | 0.5 | quantile 回帰の分位点(pinball 損失) |
| `num_threads` | 0 | スレッド数(0 = 自動) |
| `seed` | 0 | 乱数シード(同一シードでビット単位に再現) |

### アーキテクチャ

| パラメータ | 既定値 | 説明 |
|-----------|-------|------|
| `tier` | 2 | 0=純 GBDT, 1=注意読み出し, 2=+自己注意 |
| `num_tokens` | 8 | トークン数 P(上限 16) |
| `num_heads` | 2 | 注意ヘッド数 H(上限 4) |
| `attention_mode` | `"qk_leaf"` | `"qk_leaf"`(学習済 QK)/ `"score_tree"`(直接スコア) |
| `d_attn` | 4 | QK 埋め込み次元 d_a(上限 8) |
| `eta_attn` | 0.5 | 自己注意の混合率 η(Tier-2) |
| `attn_mask` | `"full"` | `"full"` / `"feature_local"`(特徴を共有するトークン間のみ自己注意。Tier-2 の過学習・メモリ抑制) |

### ツリー

| パラメータ | 既定値 | 説明 |
|-----------|-------|------|
| `token_num_leaves` | 31 | トークンツリーの最大葉数 |
| `gate_num_leaves` | 31 | ゲートツリーの最大葉数 |
| `max_depth` | -1 | 木の最大深さ(-1 = 無制限。leaf-wise 成長の深さ上限) |
| `min_data_in_leaf` | 20 | 葉の最小サンプル数(別名 `min_child_samples`) |
| `min_sum_hessian_in_leaf` | 1e-3 | 葉の最小ヘシアン量(別名 `min_child_weight`) |
| `max_bin` | 255 | ヒストグラムビン数(上限 256) |

### 正則化

| パラメータ | 既定値 | 説明 |
|-----------|-------|------|
| `lambda_l1` | 0.0 | 分割ゲインの L1 正則化(別名 `reg_alpha`) |
| `lambda_v` | 1.0 | 葉値の L2 正則化(別名 `reg_lambda`) |
| `lambda_q` | 0.1 | クエリベクトルの L2 正則化 |
| `lambda_k` | 0.1 | キーベクトルの L2 正則化 |
| `lambda_z` | 0.1 | スコアツリーパラメータの正則化 |
| `lambda_ent` | 1e-3 | 注意分布のエントロピー正則化(崩壊防止) |
| `lambda_div` | 1e-3 | ヘッド多様性正則化(cos²(Ā_h, Ā_h') を抑制、Tier-2) |
| `min_gain_to_split` | 0.0 | 分割に要求する最小ゲイン |
| `bagging_fraction` | 1.0 | 行サブサンプリング率(別名 `subsample`) |
| `bagging_freq` | 0 | バギングマスクの再抽選間隔(反復数) |
| `feature_fraction` | 1.0 | 木ごとの特徴サブサンプリング率(別名 `colsample_bytree`) |
| `early_stopping_rounds` | 0 | 検証メトリクスが改善しない場合に打ち切る反復数(`eval_set` 必須) |

早期打ち切りの監視メトリクスは回帰 = RMSE(huber は MAE、quantile は pinball)、
二値 = logloss、多クラス = logloss、ランキング = 1 − NDCG@T です。

### Phase B 内部設定

通常は変更不要です。

| パラメータ | 既定値 | 説明 |
|-----------|-------|------|
| `inner_refit_steps` | 2 | Phase B の Newton 反復数 |
| `attn_warmup` | 10 | α=1/P 固定で value のみ学習する初期反復数(実効値は min(attn_warmup, num_iterations/5)) |
| `attn_step_clip` | 1.0 | ω / b / bA のスカラー Newton ステップ上限(0=無効)。softmax の 0/1 飽和による attention collapse を防止 |
| `beta_uniform_mix` | 0.1 | Phase A のトークン重み w = (1−γ)β̂ + γ/P の γ。飢餓トークンの回復を保証 |

---

## Tier ガイド 0 1 2

| データの特性 | 推奨設定 |
|---|---|
| 特徴数 < 20 | `tier=0` または `tier=1`(`num_tokens=2〜4`) |
| 特徴数 20〜100 | `tier=1`(`num_tokens=4〜8`) |
| 特徴数 > 100 | `tier=1`(`num_tokens=8〜16`)または `tier=2` |
| サンプル数 < 500 | `tier=0` または `tier=1`(`min_data_in_leaf` を大きく) |

---

## 使用ライブラリ・著作権表示

Shimaenaga は以下のサードパーティライブラリを使用しています。

### ランタイム依存

| ライブラリ | ライセンス | 著作権 |
|---|---|---|
| [NumPy](https://github.com/numpy/numpy) | BSD 3-Clause | Copyright (c) 2005-2025, NumPy Developers |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD 3-Clause | Copyright (c) 2007-2026, The scikit-learn developers |

### ビルド時依存

| ライブラリ | ライセンス | 著作権 |
|---|---|---|
| [pybind11](https://github.com/pybind/pybind11) | BSD 3-Clause | Copyright (c) 2016 Wenzel Jakob and others |

---

## アルゴリズム参照・著作権表示

Shimaenaga は以下のプロジェクトで発表されたアルゴリズムや手法を参考に実装しています。

### XGBoost

- **Project**: https://github.com/dmlc/xgboost
- **License**: Apache 2.0
- **Reference**: Tianqi Chen and Carlos Guestrin. "XGBoost: A Scalable Tree Boosting System." KDD 2016. https://arxiv.org/abs/1603.02754
- **参考にした手法**: L1/L2 正則化付き葉の重み計算式、ヘシアンベースの分割ゲイン、最小子ノード制約

### LightGBM

- **Project**: https://github.com/microsoft/LightGBM
- **License**: MIT
- **Reference**: Guolin Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- **参考にした手法**: ヒストグラムベース分割探索・差分トリック、Leaf-wise 木成長、GOSS（勾配ベースサンプリング）、EFB（排他的特徴量バンドリング）

### CatBoost

- **Project**: https://github.com/catboost/catboost
- **License**: Apache 2.0
- **Reference**: Liudmila Prokhorenkova et al. "CatBoost: unbiased boosting with categorical features." NeurIPS 2018. https://arxiv.org/abs/1706.09516
- **参考にした手法**: Ordered Boosting（予測シフト低減）、Ordered Target Encoding（カテゴリカル特徴量）、対称木（Oblivious Tree）成長

### Gradient Boosting Machine

- **Reference**: Jerome H. Friedman. "Greedy Function Approximation: A Gradient Boosting Machine." Annals of Statistics, 2001. https://www.jstor.org/stable/2699986
- **参考にした手法**: 勾配ブースティング全体の理論的基盤(残差への逐次的な関数近似、加法モデル)

### Attention Is All You Need

- **Reference**: Ashish Vaswani et al. "Attention Is All You Need." NeurIPS 2017. https://arxiv.org/abs/1706.03762
- **参考にした手法**: スケーリング付き内積注意(scaled dot-product attention, softmax(QK^T / √d))、マルチヘッド注意 — Tier-2 自己注意およびゲート機構の設計に対応

### LambdaMART

- **Reference**: Christopher J.C. Burges. "From RankNet to LambdaRank to LambdaMART: An Overview." Microsoft Research Technical Report, 2010. https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
- **参考にした手法**: ランキング目的関数における λ 勾配(NDCG 変化量に基づくペアワイズ勾配のスケーリング)

---

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) を参照してください。
サードパーティ(pybind11)の著作権表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
