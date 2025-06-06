# QuantumLOFClassifier – Guo 2023 準拠版

このリポジトリは、以下の論文に記載されたアルゴリズムの**実装に挑戦**したものです：

> Ming-Chao Guo ら, “Quantum Algorithm for Unsupervised Anomaly Detection”
> [arXiv:2304.08710 (2023)](https://arxiv.org/abs/2304.08710)

## 📘 概要

`QuantumLOFClassifier` は、量子回路（Hadamard テスト）を用いてベクトル間の内積を推定し、その結果を基に LOF（Local Outlier Factor）を計算する、量子強化型の異常検知分類器です。

* ✅ **量子 LOF ステップ**（Qiskit 2.0.2 互換の回路を使用）
* ✅ **Hadamard テスト** による ⟨x|y⟩ の推定
* ✅ 論文の通りの LOF スコア計算
* ✅ クリーン領域・ノイズ領域向けの二段階分類器
* ⚠️ Grover ベースの最小値探索／平均計算は未実装

---

## 🧠 論文対応マッピング

| 論文セクション            | 機能                              | 実装状況  | メモ                               |             |
| ------------------ | ------------------------------- | ----- | -------------------------------- | ----------- |
| III-A Eq.(13–14)   | 振幅エンベッディング                      | ✅ 部分  | `StatePreparation` を使用、QRAM ではない |             |
| III-A Fig. 3       | Hadamard テスト ⟨x                 | y⟩    | ✅ 実装                             | 正確なアンシラ制御回路 |
| III-A Eq.(15–17)   | 内積から距離への変換                      | ✅ 実装  | 関数 \`sqrt(2 − 2⟨x                | y⟩)\` を使用   |
| III-A Step 1.6–1.7 | 量子最小値探索（Quantum Minimum Search） | ❌ 未実装 | 古典ソートで代替                         |             |
| III-B              | 量子 LRD（逆平均）                     | ❌ 未実装 | LRD は古典的に平均計算                    |             |
| III-C Eq.(18)      | LOF ≥ δ のしきい値判定                 | ✅ 実装  | LOF 計算および閾値判定を完全再現               |             |
| Eq.(2), Eq.(28)    | Grover ベースの外れ値抽出                | ❌ 未実装 | 古典的なしきい値判定で代替                    |             |

---

## 🚀 インストール

```bash
pip install quantlof
```

### 必要条件

* Python 3.8 以上
* Qiskit 2.0.2
* scikit-learn
* numpy

---

## 🧪 使い方サンプル

```python
from quantum_lof import QuantumLOFClassifier

clf = QuantumLOFClassifier(
    n_neighbors=20,
    delta=1.5,
    quantum_backend='qiskit_simulator',  # または 'ibm_cairo' などの実機バックエンド
    shots=1024,
    random_state=42
)

# 学習データ（例：X_train, y_train）で学習
clf.fit(X_train, y_train)

# 異常サンプルのインデックス取得
anomalies = clf.get_anomaly_indices()

# テストデータで予測
y_pred = clf.predict(X_test)

# クリーン部分だけの精度・F1 などを評価
acc, f1, n_clean = clf.score_clean_only(X_test, y_test)
```

---

## ⚙️ 主な機能

* ✅ Hadamard テストによる ⟨x|y⟩ の内積推定
* ✅ 論文式 (15–17) に基づくユークリッド距離の計算
* ✅ k-距離の量子的推定
* ✅ 局所到達可能密度（LRD）の計算
* ✅ LOF スコア (Eq. 18) の算出と閾値判定
* ✅ クリーン領域/ノイズ領域の二段階分類器
* ✅ サンプル数が `maxsample_for_quantum` を超えた場合の古典フェールバック

---

## 📊 Guo et al. (2023) 準拠度

| 論文セクション          | 内容                        | ステータス | 備考                     |                 |
| ---------------- | ------------------------- | ----- | ---------------------- | --------------- |
| III-A Eq.(13–14) | 振幅エンベッディング                | ✅ 部分  | `StatePreparation` で実装 |                 |
| III-A Fig. 3     | Hadamard テストによる ⟨x        | y⟩ 推定 | ✅ 実装                   | アンシラ制御回路をそのまま再現 |
| III-A Eq.(15–17) | d(x,y) = √(2 − 2⟨x        | y⟩)   | ✅ 実装                   | 距離計算ロジックを完全再現   |
| III-B            | 到達可能密度（LRD）               | ✅ 古典  | 古典的平均算出で代替             |                 |
| III-C Eq.(18)    | LOF スコアの算出およびしきい値判定       | ✅ 実装  | 論文にある LOF ≥ δ の判定を再現   |                 |
| Eq.(2)           | δ による異常判定                 | ✅ 実装  | LOF ≥ δ → 異常           |                 |
| Grover, QRAM, など | 量子最小値探索／QRAM／Grover ベース抽出 | ❌ 未実装 | 全て古典的処理に置換             |                 |

---

## 📈 可視化（任意）

LOF スコアを PCA や t-SNE などで次元削減し、検出された異常を可視化できます。

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X_proj = PCA(n_components=2).fit_transform(X)
anoms = clf.get_anomaly_indices()

plt.scatter(*X_proj.T, c='gray')
plt.scatter(*X_proj[anoms].T, c='red', label='Quantum LOF Anomalies')
plt.legend()
plt.show()
```

---

## 🛠️ API

### `QuantumLOFClassifier(...)`

| 引数                      | 説明                                                     |
| ----------------------- | ------------------------------------------------------ |
| `n_neighbors`           | LOF で使用する近傍サンプル数                                       |
| `delta`                 | LOF のしきい値 （LOF ≥ δ→異常）                                 |
| `quantum_backend`       | Qiskit バックエンド名（例: `"qiskit_simulator"`, `"ibm_cairo"`） |
| `shots`                 | Hadamard テスト時のショット数                                    |
| `maxsample_for_quantum` | 量子的距離計算を行う最大サンプル数（超える場合は古典計算にフォールバック）                  |
| `clean_model`           | クリーン領域向けの下流分類器（デフォルト: SVM）                             |
| `noise_model`           | 全サンプル向けの分類器（デフォルト: RandomForest）                       |

---

## 🤖 実装のポイント

* 内積推定には **Hadamard テスト** を利用し、`StatePreparation` の逆回路をアンシラ制御して構築
* LRD や LOF スコア自体は **古典的な平均計算** で実装
* サンプル数が `maxsample_for_quantum` を超えると、全距離を古典的に計算するフェールバックを挟む

---

## 📜 ライセンス

MIT License © 2025 [Yuto Mueller](mailto:geoyuto@gmail.com)

---

## 💡 今後の予定

* Grover ベースの最小値探索の導入
* QRAM エミュレーションの実装
* LOF 用の振幅推定（Amplitude Estimation）の導入
* 古典計算フェールバックの GPU 最適化

