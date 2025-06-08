from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

from quantlof import QuantumLOFClassifier

# ────────────────────────────────────────────────────
# 1) データ準備
# ────────────────────────────────────────────────────
# （A）ベースとなる特徴量データ 490 サンプルを生成
X_base, y_base = make_classification(
    n_samples=90,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    flip_y=0.0,       # ラベルは反転させない
    random_state=42
)

# （B）特徴量空間の外れ値サンプルを 10 件追加（たとえば平均から大きく離す）
rng = np.random.RandomState(999)
outliers = rng.normal(loc=100.0, scale=1.0, size=(10, 20))
y_outliers = np.ones(10, dtype=int)  # 値はどちらでも構わない（ここではクラス 1 に設定）

# （C）結合して 500 サンプルとする
X = np.vstack([X_base, outliers])
y = np.hstack([y_base, y_outliers])

# 学習 / テスト分割 (30% をテスト)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ────────────────────────────────────────────────────
# 2) QuantumLOFClassifier の初期化
# ────────────────────────────────────────────────────
clf = QuantumLOFClassifier(
    n_neighbors=20,
    delta=2.0,
    quantum_backend='qiskit_simulator',  # AerSimulator を使用
    # quantum_backend='ibm_cairo',       # 実機 ibm_cairo を使用する場合
    shots=10,
    random_state=42,
    maxsample_for_quantum=200,
    distance_metric="euclid"
    )

# ────────────────────────────────────────────────────
# 3) LOF による異常/クリーン判定 
# ────────────────────────────────────────────────────
clf.detect_anomalies(X_train, y_train)
print(clf.lof_scores_)
# ────────────────────────────────────────────────────
# 4) 訓練データで異常と判定されたインデックスを取得
# ────────────────────────────────────────────────────
anom_idx = clf.get_anomaly_indices()
clean_idx = clf.get_clean_indices()
print("Training anomalies indices:", anom_idx)

# ────────────────────────────────────────────────────
# 5) モデルを作成
# ────────────────────────────────────────────────────
clf.fit_models(y_train)

# ────────────────────────────────────────────────────
# 6) テストセット全体でラベル予測
# ────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# ────────────────────────────────────────────────────
# 7) テストセット中、LOF < delta (クリーン判定) のみで評価
# ────────────────────────────────────────────────────
acc_clean, f1_clean, n_clean = clf.score_clean_only(X_test, y_test)
print(f"Clean-only accuracy: {acc_clean:.4f}, F1: {f1_clean:.4f}, n_clean: {n_clean}")
