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
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestCentroid
import numpy as np

# (A) ベースデータ作成
X_base, y_base = make_classification(
    n_samples=90,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    flip_y=0.0,
    class_sep=1.5,  # クラス分離度を高めに
    random_state=42
)

# (B) クラス1の重心を求める
clf_nc = NearestCentroid()
clf_nc.fit(X_base, y_base)
class1_center = clf_nc.centroids_[1]

# (C) クラス1の近くに偽サンプルを配置（擬態型ノイズ）
rng = np.random.RandomState(999)
outliers = class1_center + rng.normal(loc=0.0, scale=0.5, size=(10, 20))
y_outliers = np.full(10, 2)  # ラベルを偽装

# (D) 結合
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
