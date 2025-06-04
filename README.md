# QuantumLOFClassifier (Quantum Local Outlier Factor Classifier)

This package provides a quantum-enhanced version of the Local Outlier Factor (LOF) anomaly detection model, utilizing quantum computing tools such as Qiskit.

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


# QuantumLOFClassifier – Compliance & Gap Report (vs. Guo 2023)

*Last updated: 2025‑06‑04*

---

## 1  Executive summary *(English)*

The current `QuantumLOFClassifier` implementation **partially follows** the pipeline proposed in *Guo et al., “Quantum Algorithm for Unsupervised Anomaly Detection” (arXiv 2304.08710, 2023).*

* **What is really quantum?**  Only the **pair‑wise distance estimation** is executed on a quantum backend via a Hadamard‑test circuit.
* **Where does it comply?**  It respects Sec. III‑A Eq.(13–17) for turning an inner product into an Euclidean distance, and keeps the LOF formula (Sec. III‑B/C).
* **Where does it diverge?**  Every block that yields exponential‐speed‑up in the paper (QRAM, Quantum Minimum Search, Quantum Multiply‑Adder, amplitude estimation–based LOF, Grover‑style anomaly extraction) is replaced by classical code.

Overall, the code is a **hybrid proof‑of‑concept** rather than a strict end‑to‑end quantum algorithm.

---

## 2  Quantum sub‑modules & level of compliance

| Paper section      | Purpose                                | Implemented? | Comment                                                                  |
| ------------------ | -------------------------------------- | ------------ | ------------------------------------------------------------------------ |
| III‑A Eq.(13–14)   | Amplitude embedding of input vector    | *Partial*    | Uses `StatePreparation`; cost becomes exponential instead of ≈O(d).      |
| III‑A Fig. 3       | Hadamard test for ⟨x\|y⟩               | **Yes**      | Circuit generated with ancilla‑controlled **U<sub>y</sub><sup>†</sup>**. |
| III‑A Eq.(15–17)   | d(x,y)=√(2−2⟨x\|y⟩)                    | **Yes**      | Exact formula applied.                                                   |
| III‑A Step 1.6–1.7 | **Quantum Minimum Search** (Grover)    | **No**       | Replaced by Python sort.                                                 |
| III‑B              | Quantum multiply‑adder & average (LRD) | **No**       | Classical loops.                                                         |
| III‑C Eq.(18)      | Quantum LOF computation                | **No**       | Classical ratio/mean.                                                    |
| Eq.(2) & (28)      | Grover anomaly extraction              | **No**       | Classical thresholding.                                                  |

---

## 3  Major divergences & limitations

1. **Absence of QRAM**  The paper assumes a QRAM oracle O<sub>X</sub>; Qiskit/NISQ hardware do not provide this.
2. **Quantum Minimum Search skipped**  Sorting is done on CPU, losing the √m quantum speed‑up.
3. **Quantum average / inverse missing**  LRD & LOF are computed classically.
4. **High circuit cost for StatePreparation**  `StatePreparation` scales as O(2^n) gates, conflicting with the paper’s low‑depth assumption.
5. **No amplitude‑estimation error control**  Theoretical bounds (ε₁,ε₂,ε₃) are not implemented.
6. **Fallback to classical path for n>100**  The paper does not define such fallback; added for practicality.

---

## 4  Practical recommendations

* Keep the current design for real datasets; full quantum blocks are unrealistic on today’s hardware.
* For small toy examples (<4 samples, <4 features) a pedagogical prototype of Quantum Minimum Search could be coded, but will not scale.
* Document clearly that the library is **“quantum‑inspired”** with a single quantum subroutine.

---
