# QuantumLOFClassifier – Fully Guo 2023-Compliant

This repository provides a **faithful implementation(as far as I can)** of the algorithm proposed in:

> Ming-Chao Guo et al., “Quantum Algorithm for Unsupervised Anomaly Detection”
> [arXiv:2304.08710 (2023)](https://arxiv.org/abs/2304.08710)

## 📘 Overview

`QuantumLOFClassifier` is a quantum-enhanced Local Outlier Factor (LOF) anomaly detection classifier, using **Hadamard-test quantum circuits** to estimate inner products between vectors, which are then used to calculate pairwise distances for LOF scoring.

* ✅ **Quantum LOF step** using Qiskit 2.0.2 compatible circuits
* ✅ **Hadamard test** for inner product ⟨x|y⟩ via ancilla-mediated circuit
* ✅ **LOF score computation** as per the original paper
* ✅ Dual downstream models for clean and noisy regions
* ⚠️ Currently lacks Grover-based quantum minimum/average steps

---

## 🧠 Algorithm Mapping to Paper

| Paper Section      | Functionality                 | Implemented? | Notes                                    |                                  |
| ------------------ | ----------------------------- | ------------ | ---------------------------------------- | -------------------------------- |
| III-A Eq.(7)   | Amplitude embedding           | ✅ Partial    | Uses `StatePreparation`, not QRAM oracle |                                  |
| III-A Fig. 3       | Hadamard test ⟨x｜y⟩           | ✅ Yes                                    | Exact ancilla-based test circuit |
| III-A Eq.(15–17)   | Distance from inner product   | ⚠️Partial       |    My implementation estimates the real part of ⟨x｜y⟩ via the Hadamard test, and uses it to compute the Euclidean distance as d(x, y) = √(2 − 2⟨x｜y⟩), assuming normalized inputs.                  |
| III-A Step 1.6–1.7 | Quantum Minimum Search        | ❌ No         | Replaced with classical sort             |                                  |
| III-B              | Quantum LRD (inverse average) | ❌ No         | Classical mean-based implementation      |                                  |
| Eq.(2), Eq.(28)    | Grover anomaly extraction     | ❌ No         | Classical threshold test                 |                                  |

---

## 🚀 Installation

```bash
pip install quantlof
```

---

## 🧪 Example Usage

```python
from quantum_lof import QuantumLOFClassifier

clf = QuantumLOFClassifier(
    n_neighbors=20,
    delta=1.5,
    quantum_backend='qiskit_simulator',  # or actual IBM backend like 'ibm_cairo'
    shots=1024,
    random_state=42
)

clf.detect_anomalies(X, y)
print(clf.lof_scores_)

anom_idx = clf.get_anomaly_indices()
clean_idx = clf.get_clean_indices()
print("Anomalies indices:", anom_idx)
print("Clean indices:", clean_idx)
```

---

## ⚙️ Core Features

* ✅ Hadamard-test for ⟨x|y⟩ inner products
* ✅ Euclidean distance via Eq. (15–17)
* ✅ k-distance via quantum estimation
* ✅ Local Reachability Density (LRD)
* ✅ LOF scores (Eq. 18) with thresholding
* ✅ Clean/noise classification downstream
* ✅ Fallback to classical when `n > maxsample_for_quantum`

---

## 🛠️ API

### `QuantumLOFClassifier(...)`

| Argument                | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| `n_neighbors`           | Number of neighbors for LOF                               |
| `delta`                 | LOF threshold (LOF ≥ δ → anomaly)                         |
| `quantum_backend`       | Qiskit backend (e.g. `"qiskit_simulator"`, `"ibm_cairo"`) |
| `shots`                 | Number of shots in Hadamard test                          |
| `maxsample_for_quantum` | Fallback threshold for classical mode                     |
| `clean_model`           | Classifier for clean samples (default: SVM)               |
| `noise_model`           | Classifier for all samples (default: RandomForest)        |

---

## 🤖 Implementation Notes

* The quantum inner product estimation uses **Hadamard test** with controlled inverse `StatePreparation`.
* Classical LOF score is maintained to allow hybrid quantum–classical behavior.
* The quantum part only replaces pairwise distance calculations.

---

## 📜 License

MIT License © 2025 [Yuto Mueller](mailto:geoyuto@gmail.com)

---

## 💡 Future Work

* Grover-based minimum search
* QRAM emulation
* Amplitude estimation for LOF
* GPU-accelerated classical fallback
