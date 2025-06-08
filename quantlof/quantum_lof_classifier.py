# quantum_lof_classifier.py
"""
Quantum-enhanced Local Outlier Factor
=====================================
Guo 2023 (arXiv:2304.08710) 参考。

* Fidelity :  √(1−F),  F=|⟨x|y⟩|²
    └─ swap      : SWAP-test（2n+1 qubits, 1 回路）

Qiskit 2.0.2 には `SwapTest` 回路は無いため、
CSWAP ループで手動構築している。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

__all__ = ["QuantumLOFClassifier"]

MAX_QUANTUM_SAMPLES: int = 100
EPS: float = 1e-12

# ---------------------------------------------------------------------------
#   Basic helpers
# ---------------------------------------------------------------------------

def _next_pow_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def _amp_embed_state(vec: np.ndarray) -> Tuple[List[float], int]:
    """Normalise and zero-pad → return (amplitudes, n_qubits)."""
    v = vec.astype(float)
    v = v / (np.linalg.norm(v) + EPS)
    dim = _next_pow_two(len(v))
    if dim != len(v):
        v = np.pad(v, (0, dim - len(v)))
    return v.tolist(), int(math.log2(dim))

def _sin2_embed_state(vec: np.ndarray) -> Tuple[List[float], int]:
    """sin² encoding: amplitudes = sin(x_i)."""
    v = np.sin(vec)
    v = v / (np.linalg.norm(v) + EPS)
    dim = _next_pow_two(len(v))
    if dim != len(v):
        v = np.pad(v, (0, dim - len(v)))
    return v.tolist(), int(math.log2(dim))

# ---------------------------------------------------------------------------
#   Fidelity helpers
# ---------------------------------------------------------------------------

def _make_swap_test(xi: np.ndarray, xj: np.ndarray) -> Tuple[QuantumCircuit, str]:
    """Construct SWAP-test manually using `initialize()` (AerSimulator safe)."""
    amps_i, nq = _sin2_embed_state(xi)
    amps_j, _  = _sin2_embed_state(xj)

    anc  = QuantumRegister(1, "anc")
    reg_a = QuantumRegister(nq, "a")
    reg_b = QuantumRegister(nq, "b")
    creg = ClassicalRegister(1, "c")
    qc = QuantumCircuit(anc, reg_a, reg_b, creg, name="SwapTest")

    # Hadamard on ancilla
    qc.h(anc[0])

    # Safe initialization instead of StatePreparation
    qc.initialize(amps_i, reg_a)
    qc.initialize(amps_j, reg_b)

    # CSWAPs
    for i in range(nq):
        qc.cswap(anc[0], reg_a[i], reg_b[i])

    # Final Hadamard and measurement
    qc.h(anc[0])
    qc.measure(anc[0], creg[0])
    return qc, "0"

def _estimate_fidelity_swap(
    backend: Union[AerSimulator, "BackendV2"],
    xi: np.ndarray,
    xj: np.ndarray,
    shots: int,
) -> float:
    qc, key0 = _make_swap_test(xi, xj)
    rand_seed = np.random.randint(1, 2**32-1)
    try:
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts()
        p0 = counts.get(key0, 0) / shots
        return 2 * p0 - 1.0  # F = 2p0 − 1
    except Exception as e:
        print("SWAP fallback used. ERROR:", e)
        inner = float(np.dot(xi, xj) / (np.linalg.norm(xi) * np.linalg.norm(xj) + EPS))
        return inner ** 2

# ---------------------------------------------------------------------------
#   Distance conversion
# ---------------------------------------------------------------------------

def _distance_from_fidelity(F: float) -> float:
    F_clip = min(max(F, 0.0), 1.0)
    return math.sqrt(max(1.0 - F_clip, 0.0))

# ---------------------------------------------------------------------------
#   QuantumLOFClassifier
# ---------------------------------------------------------------------------

class QuantumLOFClassifier(BaseEstimator, OutlierMixin, ClassifierMixin):
    """
    Quantum Local Outlier Factor with selectable distance metric.

    Parameters
    ----------
    distance_metric : {'euclid', 'fidelity'}
        'euclid'   – √(2−2⟨x|y⟩)   (Hadamard 1回)
        'fidelity' – √(1−F)        (swap)
    fidelity_method : {'hadamard2', 'swap'}
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        delta: float = 1.5,
        distance_metric: str = "euclid",
        quantum_backend: str = "qiskit_simulator",
        shots: int = 1024,
        random_state: Optional[int] = None,
        clean_model=None,
        noise_model=None,
        maxsample_for_quantum: int = MAX_QUANTUM_SAMPLES,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.delta = delta
        self.distance_metric = distance_metric.lower()
        self.quantum_backend = quantum_backend
        self.shots = shots
        self.random_state = random_state
        self.clean_model = clean_model or SVC(
            probability=True, kernel="rbf", C=1.0, random_state=random_state
        )
        self.noise_model = noise_model or RandomForestClassifier(
            n_estimators=100, random_state=random_state
        )
        self.maxsample_for_quantum = maxsample_for_quantum

        # placeholders
        self.scaler_: Optional[StandardScaler] = None
        self.X_train_: Optional[np.ndarray] = None
        self.k_distances_: Optional[np.ndarray] = None
        self.neighbor_indices_: Optional[List[List[int]]] = None
        self.lrd_scores_: Optional[np.ndarray] = None
        self.lof_scores_: Optional[np.ndarray] = None
        self.anomaly_indices_: Optional[np.ndarray] = None
        self.clean_indices_: Optional[np.ndarray] = None
        self._backend = None

    # ---------------------------------------------------------------------

    def _build_backend(self):
        if self.quantum_backend.lower() == "qiskit_simulator":
            self._backend = AerSimulator(method="statevector")
        else:
            try:
                self._backend = QiskitRuntimeService().backend(self.quantum_backend)
            except Exception:
                self._backend = AerSimulator(method="statevector")

    # --- similarity wrapper ---------------------------------------------

    def _similarity(self, xi: np.ndarray, xj: np.ndarray) -> float:
        return _estimate_fidelity_swap(self._backend, xi, xj, self.shots)

    # --- k-distance ------------------------------------------------------

    def _quantum_k_distance(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        kdist = np.zeros(n)
        # fidelity
        
        for i in range(n):
            dlist = []
            for j in range(n):
                if i == j:
                    continue
                sim = self._similarity(X[i], X[j])
                dlist.append(_distance_from_fidelity(sim))
            dlist.sort()
            kdist[i] = dlist[self.n_neighbors - 1]
        return kdist

    # ------------------------ public API ------------------------

    def detect_anomalies(self, X: np.ndarray, y: np.ndarray):
        """Compute LOF scores and split indices into anomalies / clean."""
        X, y = np.asarray(X, float), np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.X_train_ = Xs

        self._build_backend()

        if len(X) <= self.maxsample_for_quantum:
            print("Use quantum k-distance")
            self.k_distances_ = self._quantum_k_distance(Xs)
        else:
            print("Use classical k-distance")
            nbr_tmp = NearestNeighbors(n_neighbors=self.n_neighbors).fit(Xs)
            self.k_distances_ = nbr_tmp.kneighbors(Xs)[0][:, -1]

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(Xs)
        _, idx = nbrs.kneighbors(Xs)
        self.neighbor_indices_ = [row[1:].tolist() for row in idx]

        n = len(Xs)
        self.lrd_scores_ = np.zeros(n)
        for i, neigh in enumerate(self.neighbor_indices_):
            reach = [max(self.k_distances_[j], np.linalg.norm(Xs[i] - Xs[j])) for j in neigh]
            self.lrd_scores_[i] = 1.0 / (np.mean(reach) + EPS)

        self.lof_scores_ = np.zeros(n)
        for i, neigh in enumerate(self.neighbor_indices_):
            self.lof_scores_[i] = np.mean(
                [self.lrd_scores_[j] / (self.lrd_scores_[i] + EPS) for j in neigh]
            )

        is_noise = self.lof_scores_ >= self.delta
        self.anomaly_indices_, self.clean_indices_ = (
            np.where(is_noise)[0],
            np.where(~is_noise)[0],
        )
        return self

    def fit_models(self, y: np.ndarray):
        if self.X_train_ is None or self.anomaly_indices_ is None:
            raise RuntimeError("Call detect_anomalies() before fit_models().")
        Xs = self.X_train_
        mask = np.zeros(len(Xs), dtype=bool)
        mask[self.anomaly_indices_] = True

        if len(self.clean_indices_) > 0:
            self.clean_model.fit(Xs[~mask], y[~mask])
        self.noise_model.fit(Xs, y)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.detect_anomalies(X, y)
        self.fit_models(y)
        return self

    # ---------------- prediction & scoring ----------------------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X_train_)
        d, idx = nbrs.kneighbors(Xs)

        lrd_t = np.zeros(len(X))
        for i in range(len(X)):
            reach = [max(self.k_distances_[j], d[i, k]) for k, j in enumerate(idx[i])]
            lrd_t[i] = 1 / (np.mean(reach) + EPS)

        lof_t = np.zeros(len(X))
        for i in range(len(X)):
            lof_t[i] = np.mean([self.lrd_scores_[j] / (lrd_t[i] + EPS) for j in idx[i]])
        return lof_t

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        lof = self.decision_function(X)
        is_noise = lof >= self.delta
        y_hat = np.zeros(len(X), int)
        if np.any(~is_noise):
            y_hat[~is_noise] = self.clean_model.predict(Xs[~is_noise])
        if np.any(is_noise):
            y_hat[is_noise] = self.noise_model.predict(Xs[is_noise])
        return y_hat

    def score_clean_only(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
        lof = self.decision_function(X)
        mask = lof < self.delta
        if not np.any(mask):
            return 0.0, 0.0, 0
        Xc, yc = self.scaler_.transform(X[mask]), y[mask]
        yp = self.clean_model.predict(Xc)
        return accuracy_score(yc, yp), f1_score(yc, yp, average="weighted"), int(mask.sum())

    # --------------------- accessors ----------------------------

    def get_anomaly_indices(self) -> np.ndarray:
        if self.anomaly_indices_ is None:
            raise RuntimeError("detect_anomalies() must be called first.")
        return self.anomaly_indices_

    def get_clean_indices(self) -> np.ndarray:
        if self.clean_indices_ is None:
            raise RuntimeError("detect_anomalies() must be called first.")
        return self.clean_indices_


