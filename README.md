# QuantumLOFClassifier (Quantum Local Outlier Factor Classifier)

This package provides a quantum-enhanced version of the Local Outlier Factor (LOF) anomaly detection model, utilizing quantum computing tools such as Qiskit and D-Wave.

## 📘 Overview

`QuantumLOFClassifier` is an estimator based on the classical Local Outlier Factor (LOF) method, enhanced with quantum-based k-distance estimation using backends such as simulators or real quantum devices.  
> **Note:** Real quantum hardware execution is *not yet tested*.

- Quantum circuit execution via **Qiskit AerSimulator** or **IBM Quantum Runtime**
- Classical LOF concepts: k-distance, local reachability density (LRD), and LOF scores
- Anomaly detection based on a user-defined `delta` threshold
- Dual-model architecture: separate classifiers for clean and noisy data

---

## 🔍 References

This implementation is inspired by the following paper:

- Ming-Chao Guo et al., *Quantum Algorithm for Unsupervised Anomaly Detection*  
  - Section II.A: LOF definitions and thresholding (LOF(x) ≥ δ → anomaly)  
  - Sections III.A–C: Quantum distance estimation, k-distance, LRD, and LOF calculations

---

## 🚀 Example Usage

```python
from quantum_lof import QuantumLOFClassifier

clf = QuantumLOFClassifier(
    n_neighbors=20,
    delta=1.5,
    quantum_backend='qiskit_simulator',  # or 'ibm_cairo', etc.
    shots=512,
    random_state=42
)

clf.fit(X_train, y_train)
anomalies = clf.get_anomaly_indices()
y_pred = clf.predict(X_test)
acc_clean, f1_clean, n_clean = clf.score_clean_only(X_test, y_test)
