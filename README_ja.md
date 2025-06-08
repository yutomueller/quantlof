
---

# QuantumLOFClassifier – Guo 2023 論文に部分準拠した量子LOF実装

このリポジトリは、以下の論文に**着想を得た**量子異常検知アルゴリズムの実装を提供します：

> Ming-Chao Guo ほか, “Quantum Algorithm for Unsupervised Anomaly Detection”
> [arXiv:2304.08710 (2023)](https://arxiv.org/abs/2304.08710)

---

## 📘 概要

`QuantumLOFClassifier` は、**アダマールテストに基づく量子回路**を用いてベクトル間の内積を推定し、それに基づいてユークリッド距離と LOF スコアを計算する、量子拡張型の異常検知（LOF）分類器です。

* ✅ Qiskit 2.0.2 対応の量子 LOF ステップを実装
* ✅ LOF スコア計算（ただし手法は独自）
* ✅ クリーン領域とノイズ領域で異なる分類器を併用
* ⚠️ Grover 検索や量子平均などの部分は未実装

---

## 🧠 論文とのアルゴリズム対応表（Guo 2023 - arXiv:2304.08710）

| 論文セクション           | 機能                                     | 実装状況 | 補足                                                                                         |
|--------------------------|------------------------------------------|----------|----------------------------------------------------------------------------------------------|
| III-A 式(7)              | 振幅埋め込み（Amplitude embedding）     | ✅ 済     | `sin²`エンコーディングを使用：`amplitudes = sin(x_i)` によって実装。                        |
| III-A 式(11)             | SWAPテストによる内積⟨x｜y⟩の推定         | ✅ 済     | CSWAPゲートと `initialize()` による手動構築で実現。                                         |
| III-A 式(15)–(17)        | 内積からの距離計算                        | ⚠ 一部   | 正規化済み入力に対して `d(x, y) = √(1 - ⟨x｜y⟩)` を使用（振幅ベースの忠実度ではない）。     |
| III-A ステップ1.6–1.7   | 量子最小値探索                            | ❌ 未実装 | 古典的なソートにより代替。                                                                   |
| III-B                    | Quantum LRD（局所到達密度の逆数）        | ❌ 未実装 | 到達距離の平均をクラシカルに計算。                                                          |
| 式(2), 式(28)            | Groverベースの異常検出                   | ❌ 未実装 | LOFスコアに基づく閾値処理によりクラシカルに異常を判定。                                     |

---

## 🚀 インストール方法

```bash
pip install quantlof
```

---

## 🧪 使用例

```python
from quantum_lof import QuantumLOFClassifier

clf = QuantumLOFClassifier(
    n_neighbors=20,
    delta=1.5,
    quantum_backend='qiskit_simulator',  # 実機使用例: 'ibm_cairo'
    shots=1024,
    random_state=42
)

clf.detect_anomalies(X, y)
print(clf.lof_scores_)

anom_idx = clf.get_anomaly_indices()
clean_idx = clf.get_clean_indices()
print("異常点のインデックス:", anom_idx)
print("正常点のインデックス:", clean_idx)
```

---

## ⚙️ 主な機能

* ✅ アダマールテストによる ⟨x｜y⟩ 推定
* ✅ 内積から距離を計算（Eq. 15–17 相当）
* ✅ 量子的 k-近傍距離推定（Hadamardベース）
* ✅ LOF スコアと LRD のクラシカル計算
* ✅ LOF 閾値に基づく異常・正常分類
* ✅ クリーン・ノイズ領域で異なる下流分類器
* ✅ サンプル数が多い場合は自動でクラシカルにフォールバック

---

## 🛠️ API

### `QuantumLOFClassifier(...)`

| 引数                      | 説明                                                     |
| ----------------------- | ------------------------------------------------------ |
| `n_neighbors`           | LOF の近傍数                                               |
| `delta`                 | 異常とみなす LOF 閾値（LOF ≥ δ）                                 |
| `quantum_backend`       | Qiskit バックエンド（例: `"qiskit_simulator"` や `"ibm_cairo"`） |
| `shots`                 | アダマールテストのショット数                                         |
| `maxsample_for_quantum` | クラシカル処理に切り替えるサンプル数の閾値                                  |
| `clean_model`           | クリーンなサンプルに使う分類器（デフォルト: SVM）                            |
| `noise_model`           | 全体に使う分類器（デフォルト: ランダムフォレスト）                             |

---

## 🤖 実装メモ

* 内積推定には `StatePreparation` の逆行列を制御付きで使う標準的アダマールテスト回路を使用。
* クラシカルな LOF 計算を保持しつつ、距離計算部分のみ量子的に置き換え。
* ハイブリッド型異常検知の設計（クラシカルとの併用）。

---

## 📜 ライセンス

MIT License © 2025 [Yuto Mueller](mailto:geoyuto@gmail.com)

---

## 💡 今後の開発予定

* Grover による k-最小探索の導入
* QRAM のエミュレーション
* 振幅推定による LOF スコアの完全量子化
* クラシカル部分の GPU 高速化対応

---


