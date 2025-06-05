# QuantumLOFClassifier（量子局所外れ値因子分類器）

このパッケージは、Qiskit の量子ツールを活用した、量子拡張版 Local Outlier Factor（LOF）異常検知モデルを提供します。

## 📘 概要

`QuantumLOFClassifier` は、局所外れ値因子（LOF）ベースの異常検知を行う推定器で、k-距離推定に量子バックエンド（シミュレータや実機）を用いることが可能です。
実機は未テストです。

- **Qiskit AerSimulator** または **IBM Quantum Runtime** による量子回路実行
- 古典的な LOF 手法：k-距離、局所到達可能密度（LRD）、LOFスコア
- deltaしきい値による異常検知
- 正常・異常データ用の2種の分類器を搭載

---

## 🔍 参考文献

この実装は以下の論文に着想を得ています：

- Ming-Chao Guo ほか, 「Quantum Algorithm for Unsupervised Anomaly Detection」  
  - Section II.A: LOF定義としきい値（LOF(x) ≥ δ → 異常）
  - Section III.A–C: 量子距離推定・k距離・LRD・LOF計算方法

---

## 🚀 使用例

```python
from quantum_lof import QuantumLOFClassifier

clf = QuantumLOFClassifier(
    n_neighbors=20,
    delta=1.5,
    quantum_backend='qiskit_simulator',  # または 'ibm_cairo' など
    shots=512,
    random_state=42
)

clf.fit(X_train, y_train)
anomalies = clf.get_anomaly_indices()
y_pred = clf.predict(X_test)
acc_clean, f1_clean, n_clean = clf.score_clean_only(X_test, y_test)
```

# 量子LOF分類器 ― 準拠状況と課題レポート（日本語）

## 1  概要

現在の `QuantumLOFClassifier` は、*Guo 2023* のアルゴリズムを **部分的に** 実装したハイブリッド版です。

* **量子になっている部分** : ペア間距離を求める **Hadamard テスト**（内積→距離変換）のみ。
* **論文と一致する点** : Sec. III‑A Eq.(13–17) の数式と LOF の定義（Sec. III‑B/C）。
* **論文と異なる点** : QRAM、Grover ベースの最小探索、量子乗算器・逆数回路、量子カウントによる異常抽出など、指数的高速化を生むブロックはすべてクラシカル置換。

## 2  量子サブモジュールと準拠度

| 論文箇所               | 役割            | 実装状況 | コメント                                       |
| ------------------ | ------------- | ---- | ------------------------------------------ |
| III‑A Eq.(13–14)   | 振幅エンコード       | △    | `StatePreparation` で代替。深さは指数。              |
| III‑A Fig. 3       | Hadamard テスト  | ○    | ancilla 制御付き U<sub>y</sub><sup>†</sup> 採用。 |
| III‑A Eq.(15–17)   | 距離計算          | ○    | 数式どおり。                                     |
| III‑A Step 1.6–1.7 | 量子最小探索        | ×    | Python ソートに置換。                             |
| III‑B              | 到達可能距離・平均     | ×    | for ループで計算。                                |
| III‑C Eq.(18)      | LOF 計算        | ×    | クラシカル比率平均。                                 |
| Eq.(2)             | Grover での異常抽出 | ×    | 単純閾値比較。                                    |

## 3  主な逸脱点・制約

1. **QRAM 不在**（O<sub>X</sub> を実装不可）。
2. **量子最小探索なし** ― √m 高速化が失われる。
3. **量子平均・逆数回路なし** ― LRD/LOF を古典計算。
4. **StatePreparation の回路深さが指数オーダー**。
5. **振幅推定誤差制御（ε パラメータ）の未実装**。
6. **n>100 でクラシカルフォールバック** ― 論文未定義。

## 4  実用上の提案

* 現状の「距離推定だけ量子化」方針を維持する。
* 教材としては、少サンプル・少特徴で QMS/QAE を模擬実装し、理論値と突き合わせるのが現実的。
* 「量子インスパイアド（距離のみ量子）」と明示する。
