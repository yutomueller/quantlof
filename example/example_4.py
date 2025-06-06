from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.decomposition import PCA

from quantlof import QuantumLOFClassifier

# ────────────────────────────────────────────────────
# 1) データ準備
# ────────────────────────────────────────────────────
# （A）ベースとなる特徴量データ 490 サンプルを生成
import numpy as np
from sklearn.datasets import load_digits


# Load digits
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# Base data
X_base = X_digits[:90]
y_base = y_digits[:90]

# Inject noise
rng = np.random.RandomState(42)
outliers = X_base[:10] + rng.normal(loc=1.5, scale=0.5, size=X_base[:10].shape)
y_outliers = np.full(10, -1)

# Merge
X = np.vstack([X_base, outliers])
y = np.hstack([y_base, y_outliers])

# ────────────────────────────────────────────────────
# 2) QuantumLOFClassifier の初期化
# ────────────────────────────────────────────────────
clf = QuantumLOFClassifier(
    n_neighbors=20,
    delta=1.5,
    quantum_backend='qiskit_simulator',  # AerSimulator を使用
    # quantum_backend='ibm_cairo',       # 実機 ibm_cairo を使用する場合
    shots=1024,
    random_state=42,
    maxsample_for_quantum=100
)

# ────────────────────────────────────────────────────
# 3) LOF による異常/クリーン判定 
# ────────────────────────────────────────────────────
clf.detect_anomalies(X, y)
print(clf.lof_scores_)
# ────────────────────────────────────────────────────
# 4) 訓練データで異常と判定されたインデックスを取得
# ────────────────────────────────────────────────────
anom_idx = clf.get_anomaly_indices()
clean_idx = clf.get_clean_indices()
print("Anomalies indices:", anom_idx)
print("Clean indices:", clean_idx)
