import pytest
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from quantlof import QuantumLOFClassifier

# -------------------------------
# データセット定義
# -------------------------------
dataset_generators = {
    'linear': lambda seed: make_classification(n_samples=300, n_features=20, n_informative=15, n_redundant=5, random_state=seed),
    'moons': lambda seed: make_moons(n_samples=300, noise=0.05, random_state=seed),
    'circles': lambda seed: make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=seed),
}

@pytest.mark.parametrize("dataset_name", list(dataset_generators.keys()))
@pytest.mark.parametrize("seed", [10, 11])
def test_models_on_datasets(dataset_name, seed):
    X, y = dataset_generators[dataset_name](seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=seed),
        'SVM': SVC(kernel='rbf', probability=True, random_state=seed),
        'QuantumLOF': QuantumLOFClassifier(
            n_neighbors=20,
            quantum_backend='qiskit_simulator',
            shots=512,
            random_state=seed,
            clean_model=SVC(kernel='rbf', probability=True, random_state=seed),
            noise_model=RandomForestClassifier(n_estimators=100, random_state=seed),
            delta=1.5
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"[{dataset_name}] {name} | Acc={acc:.3f}, F1={f1:.3f}")
        assert acc > 0.1, f"{name} accuracy too low ({acc:.3f})"
        assert f1 > 0.1, f"{name} F1-score too low ({f1:.3f})"

