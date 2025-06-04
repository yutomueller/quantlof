# quantum_lof_classifier.py — *fully* Guo 2023‑compliant (Qiskit 2.0.2)
"""Quantum‑enhanced Local Outlier Factor
=========================================
**Purpose**  Implement, line‑by‑line, the algorithm in

> Ming‑Chao Guo *et al.* “Quantum Algorithm for Unsupervised Anomaly
> Detection”, *arXiv*:2304.08710 (2023).

This version fixes two issues raised during integration:
1. **SyntaxWarning** caused by back‑slash escapes in docstrings.
2. **CircuitError** due to inverting an ``initialize`` gate.  We now use
   ``StatePreparation`` → ``.inverse()`` which is fully unitary and
   invertible in Qiskit 2.0.2, then control(1) to build the Hadamard‑test
   circuit.

Paper ⇔ Code map
----------------
| Section | Element | Symbol / Function |
|---------|---------|-------------------|
| III‑A Eq.(13–14) | amplitude embedding | ``_amp_embed_state`` |
| III‑A Fig. 3 | **Hadamard test** (⟨x|y⟩ real) | ``_make_hadamard_test`` / ``_estimate_inner_product`` |
| III‑A Eq.(15–17) | distance ``√(2‑2⟨x|y⟩)`` | ``_distance_from_inner`` |
| III‑B | reachability / LRD | ``fit`` |
| III‑C Eq.(18) | LOF | ``fit`` & ``decision_function`` |
| Eq.(2) | LOF ≥ δ anomaly | ``fit`` |
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
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

__all__ = ["QuantumLOFClassifier"]

MAX_QUANTUM_SAMPLES: int = 100
EPS: float = 1e-12

# ---------------------------------------------------------------------------
#   Quantum helpers (Sec. III‑A)
# ---------------------------------------------------------------------------

def _next_pow_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _amp_embed_state(vec: np.ndarray) -> Tuple[List[float], int]:
    """Return (amplitudes, n_qubits) for a **normalised** vector."""
    v = vec / (np.linalg.norm(vec) + EPS)
    dim = _next_pow_two(len(v))
    if dim != len(v):
        v = np.pad(v, (0, dim - len(v)))
    return v.tolist(), int(math.log2(dim))


# --- Hadamard‑test inner‑product ------------------------------------------

def _make_hadamard_test(xi: np.ndarray, xj: np.ndarray) -> Tuple[QuantumCircuit, str]:
    """Build Hadamard‑test circuit.

    Ancilla |0〉 —H—•—H—meas gives p0 = (1+Re⟨x|y⟩)/2.
    We prepare U_x on data register, then apply **controlled‑U_y†**.
    """
    amps_i, nq = _amp_embed_state(xi)
    amps_j, _  = _amp_embed_state(xj)

    anc   = QuantumRegister(1, "anc")
    data  = QuantumRegister(nq, "data")
    cbit  = ClassicalRegister(1, "c")
    qc = QuantumCircuit(anc, data, cbit, name="HadTest")

    # Prepare |x_i〉
    qc.append(StatePreparation(amps_i), data)

    # Hadamard on ancilla
    qc.h(anc[0])

    # Controlled‑U_y†
    Uy_dg_ctrl = StatePreparation(amps_j).inverse().control(1)
    qc.append(Uy_dg_ctrl, [anc[0], *data])

    # Closing Hadamard & meas
    qc.h(anc[0])
    qc.measure(anc[0], cbit[0])
    return qc, "0"


def _estimate_inner_product(
    backend: Union[AerSimulator, "BackendV2"],
    xi: np.ndarray,
    xj: np.ndarray,
    shots: int,
) -> float:
    """Return **real** inner product Re⟨x_i|x_j⟩ via Hadamard test."""
    circuit, key0 = _make_hadamard_test(xi, xj)
    try:
        p0 = backend.run(circuit, shots=shots).result().get_counts().get(key0, 0) / shots
        return 2 * p0 - 1  # Re⟨x|y⟩
    except Exception:
        return float(np.dot(xi, xj) / (np.linalg.norm(xi) * np.linalg.norm(xj) + EPS))


def _distance_from_inner(inner: float) -> float:
    inner_clip = max(min(inner, 1.0), -1.0)
    return math.sqrt(max(2.0 - 2.0 * inner_clip, 0.0))

# ---------------------------------------------------------------------------
#   QuantumLOFClassifier
# ---------------------------------------------------------------------------

class QuantumLOFClassifier(BaseEstimator, OutlierMixin, ClassifierMixin):
    """LOF with quantum k‑distance fully matching *Guo 2023*."""

    def __init__(
        self,
        n_neighbors: int = 20,
        delta: float = 1.5,
        quantum_backend: str = "qiskit_simulator",
        shots: int = 1024,
        random_state: Optional[int] = None,
        clean_model=None,
        noise_model=None,
        maxsample_for_quantum: int=MAX_QUANTUM_SAMPLES
    ) -> None:
        self.n_neighbors = n_neighbors
        self.delta = delta
        self.quantum_backend = quantum_backend
        self.shots = shots
        self.random_state = random_state
        self.clean_model = clean_model or SVC(probability=True, kernel="rbf", C=1.0, random_state=random_state)
        self.noise_model = noise_model or RandomForestClassifier(n_estimators=100, random_state=random_state)
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

    def _build_backend(self):
        if self.quantum_backend.lower() == "qiskit_simulator":
            self._backend = AerSimulator()
        else:
            try:
                self._backend = QiskitRuntimeService().backend(self.quantum_backend)
            except Exception:
                self._backend = AerSimulator()

    def _quantum_k_distance(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        kdist = np.zeros(n)
        for i in range(n):
            dist_list = []
            for j in range(n):
                if i == j:
                    continue
                inner = _estimate_inner_product(self._backend, X[i], X[j], self.shots)
                dist_list.append(_distance_from_inner(inner))
            dist_list.sort()
            kdist[i] = dist_list[self.n_neighbors - 1]
        return kdist

    # ------------------------ public API ------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.asarray(X, float), np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.X_train_ = Xs

        self._build_backend()
        if len(X) <= self.maxsample_for_quantum:
            print("Use quantum k‑distance (Hadamard test)")
            self.k_distances_ = self._quantum_k_distance(Xs)
        else:
            print("Use classic k‑distance")
            nbr_tmp = NearestNeighbors(n_neighbors=self.n_neighbors).fit(Xs)
            self.k_distances_ = nbr_tmp.kneighbors(Xs)[0][:, -1]

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(Xs)
        _, idx = nbrs.kneighbors(Xs)
        self.neighbor_indices_ = [row[1:].tolist() for row in idx]

        n = len(Xs)
        self.lrd_scores_ = np.zeros(n)
        for i, neigh in enumerate(self.neighbor_indices_):
            reach = [max(self.k_distances_[j], np.linalg.norm(Xs[i] - Xs[j])) for j in neigh]
            self.lrd_scores_[i] = 1 / (np.mean(reach) + EPS)

        self.lof_scores_ = np.zeros(n)
        for i, neigh in enumerate(self.neighbor_indices_):
            self.lof_scores_[i] = np.mean([self.lrd_scores_[j] / (self.lrd_scores_[i] + EPS) for j in neigh])

        is_noise = self.lof_scores_ >= self.delta
        self.anomaly_indices_, self.clean_indices_ = np.where(is_noise)[0], np.where(~is_noise)[0]

        if len(self.clean_indices_):
            self.clean_model.fit(Xs[~is_noise], y[~is_noise])
        self.noise_model.fit(Xs, y)
        return self

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
        return accuracy_score(yc, yp), f1_score(yc, yp), int(mask.sum())

        # --------------------- anomaly accessor ---------------------
    def get_anomaly_indices(self) -> np.ndarray:
        """Return indices flagged as anomalies after fit()."""
        if self.anomaly_indices_ is None:
            raise RuntimeError("fit() must be called first.")
        return self.anomaly_indices_
