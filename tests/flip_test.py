import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from quantlof import QuantumLOFClassifier


@pytest.mark.parametrize("noise", [0.01, 0.05, 0.1, 0.2])
@pytest.mark.parametrize("seed", [0, 1])
def test_quantumlof_vs_classic_models(noise, seed):
    # データ生成
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        flip_y=noise,
        random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    # モデル準備
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    svm = SVC(kernel='rbf', probability=True, random_state=seed)
    qlof = QuantumLOFClassifier(
        n_neighbors=20,
        quantum_backend='qiskit_simulator',
        shots=512,
        random_state=seed,
        clean_model=svm,
        noise_model=rf,
        delta=1.5
    )

    # 学習
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    qlof.fit(X_train, y_train)

    # 評価
    for name, model in [("RandomForest", rf), ("SVM", svm), ("QuantumLOF", qlof)]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"[{name}] noise={noise}, seed={seed} → Acc={acc:.3f}, F1={f1:.3f}")
        assert acc > 0.1, f"{name} accuracy too low: {acc}"
        assert f1 > 0.1, f"{name} F1 too low: {f1}"


