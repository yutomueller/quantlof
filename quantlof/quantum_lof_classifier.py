"""
quantum_lof_classifier.py

This module provides a QuantumLOFClassifier that performs Local Outlier Factor (LOF) anomaly detection
using quantum-enhanced k-distance estimation. It supports both AerSimulator and IBMQ Runtime backends.

References:
    Ming-Chao Guo et al., “Quantum Algorithm for Unsupervised Anomaly Detection” (QError.pdf)
    - Section II.A: LOF definitions and anomaly threshold (LOF(x) ≥ δ => anomaly)
    - Section III.A–C: Quantum distance estimation, k-distance, LRD, and LOF computations.

Usage:
    >>> from quantlof import QuantumLOFClassifier
    >>> clf = QuantumLOFClassifier(
    ...     n_neighbors=20,
    ...     delta=1.5,
    ...     quantum_backend='qiskit_simulator',  # Or 'ibm_cairo'
    ...     shots=512,
    ...     random_state=42
    ... )
    >>> clf.fit(X_train, y_train)
    >>> anomalies = clf.get_anomaly_indices()
    >>> y_pred = clf.predict(X_test)
    >>> acc_clean, f1_clean, n_clean = clf.score_clean_only(X_test, y_test)
"""

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_ibm_runtime import QiskitRuntimeService


class QuantumLOFClassifier(BaseEstimator, OutlierMixin, ClassifierMixin):
    """
    A LOF (Local Outlier Factor) classifier that uses quantum-enhanced k-distance estimation via
    either AerSimulator or IBMQ Runtime. After computing LOF scores, samples with LOF ≥ delta
    are treated as anomalies (noise), and two downstream models are trained:
      - clean_model on LOF < delta samples
      - noise_model on all samples

    Attributes:
        n_neighbors (int): Number of neighbors for k-distance.
        delta (float): Threshold for LOF score to mark anomalies.
        quantum_backend (str): 'qiskit_simulator' for AerSimulator or IBMQ backend name for Qiskit Runtime.
        shots (int): Number of shots per quantum circuit execution.
        random_state (int or None): Random seed passed to downstream models.
        clean_model: Classifier for LOF < delta region. Defaults to SVC if None.
        noise_model: Classifier for LOF ≥ delta region. Defaults to RandomForest if None.
        scaler_ (StandardScaler): Fitted scaler for input features.
        X_train_ (np.ndarray): Scaled training feature matrix.
        k_distances_ (np.ndarray): Computed k-distance for each training sample.
        neighbor_indices_ (list of lists): Indices of k-nearest neighbors for each training sample.
        lrd_scores_ (np.ndarray): Local reachability density for each training sample.
        lof_scores_ (np.ndarray): LOF score for each training sample.
        threshold_ (float): Set to delta, used for anomaly decision.
        anomaly_indices_ (np.ndarray): Indices in training set marked as anomalies (LOF ≥ delta).
        clean_indices_ (np.ndarray): Indices in training set marked as clean (LOF < delta).
        _quantum_backend: Backend instance (AerSimulator or IBMQ Runtime backend) for quantum circuits.
        _runtime_service (QiskitRuntimeService): Service object for IBMQ Runtime.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        delta: float = 1.5,
        quantum_backend: str = 'qiskit_simulator',
        shots: int = 1024,
        random_state: int = None,
        clean_model=None,
        noise_model=None,
    ):
        """
        Initializes the QuantumLOFClassifier.

        Args:
            n_neighbors (int): Number of neighbors for k-distance.
            delta (float): LOF threshold to classify anomalies.
            quantum_backend (str):
                - 'qiskit_simulator' to use AerSimulator,
                - otherwise interpreted as IBMQ Runtime backend name.
            shots (int): Number of shots per quantum circuit.
            random_state (int or None): Random seed for downstream models.
            clean_model: Classifier for clean region (LOF < delta). If None, uses SVC.
            noise_model: Classifier for noisy region (LOF ≥ delta). If None, uses RandomForest.
        """
        self.n_neighbors = n_neighbors
        self.delta = delta
        self.quantum_backend = quantum_backend
        self.shots = shots
        self.random_state = random_state

        # Avoid using "or" with estimator defaults to prevent __len__ issues.
        if clean_model is None:
            self.clean_model = SVC(probability=True, kernel='rbf', C=1.0, random_state=random_state)
        else:
            self.clean_model = clean_model

        if noise_model is None:
            self.noise_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            self.noise_model = noise_model

        self.scaler_ = None
        self.X_train_ = None
        self.k_distances_ = None
        self.neighbor_indices_ = None
        self.lrd_scores_ = None
        self.lof_scores_ = None
        self.threshold_ = None

        self.anomaly_indices_ = None
        self.clean_indices_ = None

        self._quantum_backend = None
        self._runtime_service = None

    def _build_quantum_backend(self):
        """
        Builds the quantum backend based on the `quantum_backend` argument.

        - If `quantum_backend == 'qiskit_simulator'`, sets self._quantum_backend = AerSimulator().
        - Otherwise, attempts to load an IBMQ Runtime backend with that name. Falls back to AerSimulator
          on any failure.

        Requires that QiskitRuntimeService.save_account(...) was called previously if using IBMQ.
        """
        if isinstance(self.quantum_backend, str) and self.quantum_backend.lower() == 'qiskit_simulator':
            self._quantum_backend = AerSimulator()
        else:
            try:
                self._runtime_service = QiskitRuntimeService()
                self._quantum_backend = self._runtime_service.backend(self.quantum_backend)
            except Exception:
                self._quantum_backend = AerSimulator()

    def _run_quantum_circuit(self, qc: QuantumCircuit) -> dict:
        """
        Executes a 1-qubit quantum circuit (Ry + measure) on the chosen backend and returns counts.

        Args:
            qc (QuantumCircuit): A simple 1-qubit circuit with one measurement.

        Returns:
            dict: Measurement counts, e.g., {'0': count0, '1': count1}. Returns {} if execution fails.
        """
        if self._quantum_backend is None:
            return {}

        try:
            job = self._quantum_backend.run(qc, shots=self.shots)
            result = job.result()
            return result.get_counts()
        except Exception:
            return {}

    def _estimate_pairwise_distance(self, x_i: np.ndarray, x_t: np.ndarray, C: float) -> float:
        """
        Approximates the distance ||x_i - x_t|| using a 1-qubit quantum circuit.

        Following QError.pdf Sec. III.A:
            1) Compute dist_raw = ||x_i - x_t|| classically.
            2) Let p_target = dist_raw / C, clipped to [0,1].
            3) θ = 2 * arcsin(sqrt(p_target)).
            4) Prepare a 1-qubit circuit: qc.ry(θ) -> measure.
            5) Run on backend to get counts; p_est = counts['0']/shots.
            6) Return dist_est = p_est * C.
            7) If any error or empty counts, return dist_raw.

        Args:
            x_i (np.ndarray): Feature vector of sample i (scaled).
            x_t (np.ndarray): Feature vector of sample t (scaled).
            C (float): Maximum pairwise distance among all samples.

        Returns:
            float: Estimated distance via quantum circuit or classical distance if quantum fails.
        """
        dist_raw = float(np.linalg.norm(x_i - x_t))
        p_target = min(max(dist_raw / C, 0.0), 1.0)
        safe_p = min(p_target, 1.0 - 1e-12)
        theta = 2.0 * np.arcsin(np.sqrt(safe_p))

        qr = QuantumRegister(1, name='q')
        cr = ClassicalRegister(1, name='c')
        qc = QuantumCircuit(qr, cr)
        qc.ry(theta, qr[0])
        qc.measure(qr[0], cr[0])

        counts = self._run_quantum_circuit(qc)
        if not counts:
            return dist_raw

        count_zero = counts.get('0', 0)
        p_est = count_zero / self.shots
        dist_est = p_est * C
        return dist_est

    def _quantum_k_distance(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Computes each sample's k-distance using quantum-enhanced estimation.

        Algorithm (QError.pdf Sec. III.A):
            1) Compute classical pairwise distances to find C = max_{i<j} ||X_i - X_j||.
            2) For each sample i:
               a) For each sample t != i, call _estimate_pairwise_distance(X_i, X_t, C) -> est_dists[t].
               b) Remove the i-th entry, sort the remaining distances, pick the (n_neighbors)-th smallest.
            3) Return array of length n_samples, where element i is k-distance for sample i.

        Args:
            X_scaled (np.ndarray): Scaled training data of shape (n_samples, n_features).

        Returns:
            np.ndarray: 1D array of k-distances for each training sample.
        """
        n_samples = X_scaled.shape[0]

        # (1) Compute maximum classical distance C
        C = 0.0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = float(np.linalg.norm(X_scaled[i] - X_scaled[j]))
                if d > C:
                    C = d
        if C <= 1e-12:
            C = 1.0

        # (2) Quantum-enhanced estimation of distances
        k_dists = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            est_dists = np.zeros(n_samples, dtype=float)
            for t in range(n_samples):
                if i == t:
                    est_dists[t] = 0.0
                else:
                    est_dists[t] = self._estimate_pairwise_distance(
                        X_scaled[i], X_scaled[t], C
                    )
            # (3) Exclude i, sort, pick the n_neighbors-th smallest
            non_zero = np.delete(est_dists, i)
            sorted_nn = np.sort(non_zero)
            idx = self.n_neighbors - 1
            k_dists[i] = sorted_nn[idx] if idx < len(sorted_nn) else sorted_nn[-1]

        return k_dists

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the QuantumLOFClassifier on the provided data.

        Steps:
            1. Scale features with StandardScaler.
            2. Build the quantum backend (AerSimulator or IBMQ Runtime).
            3. If n_samples ≤ 64 and quantum backend is available:
               a. Compute k-distance via _quantum_k_distance.
               b. Get neighbor indices via classical k-NN with n_neighbors + 1.
            4. Else:
               a. Compute k-distance and neighbor indices via classical k-NN.
            5. Compute local reachability density (LRD) and LOF scores classically.
            6. Set threshold_ = delta and mark anomalies: LOF ≥ delta.
               - Store anomaly_indices_ and clean_indices_.
            7. Fit clean_model on LOF < delta samples.
            8. Fit noise_model on all samples.

        Args:
            X (np.ndarray): Training feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Training labels/targets of length n_samples.

        Returns:
            self: Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, _ = X.shape

        # (1) Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.X_train_ = X_scaled

        # (2) Build quantum backend
        self._build_quantum_backend()

        # (3) Compute k-distance & neighbor indices
        use_quantum = (self._quantum_backend is not None) and (n_samples <= 64)
        if use_quantum:
            self.k_distances_ = self._quantum_k_distance(X_scaled)
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='auto').fit(X_scaled)
            _, indices = nbrs.kneighbors(X_scaled)
            self.neighbor_indices_ = [indices[i, 1:].tolist() for i in range(n_samples)]
        else:
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='auto').fit(X_scaled)
            distances, indices = nbrs.kneighbors(X_scaled)
            self.neighbor_indices_ = [indices[i, 1:].tolist() for i in range(n_samples)]
            self.k_distances_ = np.array([distances[i, -1] for i in range(n_samples)], dtype=float)

        # (4) Compute local reachability density (LRD) and LOF scores
        lrd_scores = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            neighbors = self.neighbor_indices_[i]
            reach_dists = [
                max(self.k_distances_[j], float(np.linalg.norm(X_scaled[i] - X_scaled[j])))
                for j in neighbors
            ]
            lrd_scores[i] = 1.0 / (np.mean(reach_dists) + 1e-12)
        self.lrd_scores_ = lrd_scores

        lof_scores = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            neighbors = self.neighbor_indices_[i]
            lrd_i = self.lrd_scores_[i]
            ratios = [self.lrd_scores_[j] / (lrd_i + 1e-12) for j in neighbors]
            lof_scores[i] = np.mean(ratios)
        self.lof_scores_ = lof_scores

        # (5) Threshold LOF to identify anomalies
        self.threshold_ = self.delta
        is_noise = self.lof_scores_ >= self.threshold_

        # (6) Store anomaly & clean indices
        self.anomaly_indices_ = np.where(is_noise)[0]
        self.clean_indices_ = np.where(~is_noise)[0]

        # (7) Fit downstream models
        X_clean = X_scaled[~is_noise]
        y_clean = y[~is_noise]
        if len(X_clean) > 0:
            self.clean_model.fit(X_clean, y_clean)
        self.noise_model.fit(X_scaled, y)

        return self

    def get_anomaly_indices(self) -> np.ndarray:
        """
        Returns the indices of training samples marked as anomalies (LOF ≥ delta).

        Returns:
            np.ndarray: Array of indices of anomalous training samples.

        Raises:
            RuntimeError: If called before fit().
        """
        if self.anomaly_indices_ is None:
            raise RuntimeError("fit() must be called before get_anomaly_indices().")
        return self.anomaly_indices_

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Computes LOF scores for new data X using classical LOF calculation.

        Algorithm (QError.pdf Sec. II.A):
            1. Scale X using fitted scaler.
            2. For each test sample i:
               a. Find its k neighbors in training data (classical k-NN with n_neighbors).
               b. Compute reachability distances: max(k_distance[j], ||x_i - x_train[j]||).
               c. Compute LRD_i = 1 / (mean of reachability distances).
            3. For each test sample i:
               a. LOF_i = mean( LRD_neighbor / LRD_i ) over its neighbors.

        Args:
            X (np.ndarray): Test features of shape (n_test_samples, n_features).

        Returns:
            np.ndarray: LOF scores array of length n_test_samples.
        """
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler_.transform(X)
        n_test = X_scaled.shape[0]

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto').fit(self.X_train_)
        distances, indices = nbrs.kneighbors(X_scaled)

        lof_test = np.zeros(n_test, dtype=float)
        lrd_test = np.zeros(n_test, dtype=float)

        for i in range(n_test):
            neigh_idx = indices[i].tolist()
            neigh_dists = distances[i].tolist()
            reach_dists = [max(self.k_distances_[j], d_ij) for j, d_ij in zip(neigh_idx, neigh_dists)]
            lrd_i = 1.0 / (np.mean(reach_dists) + 1e-12)
            lrd_test[i] = lrd_i

        for i in range(n_test):
            neighbors = indices[i].tolist()
            lrd_i = lrd_test[i]
            ratios = [self.lrd_scores_[j] / (lrd_i + 1e-12) for j in neighbors]
            lof_test[i] = np.mean(ratios)

        return lof_test

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for new data X. 
        - LOF(x) < delta => use clean_model
        - LOF(x) ≥ delta => use noise_model

        Args:
            X (np.ndarray): Test features of shape (n_test_samples, n_features).

        Returns:
            np.ndarray: Predicted labels for X.
        """
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler_.transform(X)

        lof_vals = self.decision_function(X)
        is_noise = lof_vals >= self.threshold_

        labels = np.zeros(X_scaled.shape[0], dtype=int)
        clean_idxs = ~is_noise
        noise_idxs = is_noise

        if np.any(clean_idxs):
            labels[clean_idxs] = self.clean_model.predict(X_scaled[clean_idxs])
        if np.any(noise_idxs):
            labels[noise_idxs] = self.noise_model.predict(X_scaled[noise_idxs])

        return labels

    def score_clean_only(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
        """
        Evaluates the clean_model only on samples that LOF classifies as clean (LOF < delta).

        Args:
            X_test (np.ndarray): Test features (n_test_samples, n_features).
            y_test (np.ndarray): True labels for X_test.

        Returns:
            tuple:
                accuracy (float): Accuracy on LOF < delta samples.
                f1 (float): F1-score on LOF < delta samples.
                n_clean (int): Number of samples classified as clean by LOF.

        Notes:
            Returns (0.0, 0.0, 0) if no test samples are classified as clean.
        """
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test)
        lof_vals = self.decision_function(X_test)
        is_clean = lof_vals < self.threshold_

        if not np.any(is_clean):
            return 0.0, 0.0, 0

        X_clean_test = X_test[is_clean]
        y_clean_test = y_test[is_clean]

        y_pred_clean = self.clean_model.predict(self.scaler_.transform(X_clean_test))
        accuracy = accuracy_score(y_clean_test, y_pred_clean)
        f1 = f1_score(y_clean_test, y_pred_clean)

        return accuracy, f1, int(np.sum(is_clean))

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Convenience method: fit the model and then predict on the same data.

        Args:
            X (np.ndarray): Features of shape (n_samples, n_features).
            y (np.ndarray): True labels of length n_samples.

        Returns:
            np.ndarray: Predicted labels for X.
        """
        self.fit(X, y)
        return self.predict(X)






