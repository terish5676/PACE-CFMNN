# Document 03: How We Did It, What We Used & Code Implementation Deep-Dive

## 1. Executive Summary in Simple Human Terms

If an engineer asks: *"How did you build this, what tools did you use, and how do I run it?"*, here is the straightforward blueprint:

- **We Avoided Bloated Hardware Dependencies:** Unlike deep neural networks that require giant, power-hungry Nvidia GPUs, proprietary CUDA drivers, and complex multi-gigabyte PyTorch/TensorFlow runtimes, PACE-CFMNN runs entirely on **standard, energy-efficient CPU hardware**.
- **We Used High-Performance Vectorized Math:** The core algorithms are built with NumPy and SciPy, compiled to take advantage of modern processor instruction sets (AVX-512 and AVX2). This means calculations happen in parallel directly inside the CPU registers.
- **We Rigorously Benchmarked Against the Best in the World:** To test our claims fairly, we integrated Microsoft's Explainable Boosting Machines (InterpretML), Microsoft's DiCE counterfactual library, and Scikit-learn's Random Forest and kNN baselines.
- **Clean, Modular Code Architecture:** The repository is split cleanly into five logical folders: `cfmnn/` (base models), `explainers/` (the PACE explanation algorithms), `repair/` (self-repair and certified gates), `trust/` (error lie detectors), and `baselines/` (external comparison models).

```
+---------------------------------------------------------------------------------------------------------+
|                                  THE PACE-CFMNN ENGINEERING TECH STACK                                  |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   +-----------------------------------------------------------------------------------------------+     |
|   | APPLICATION & BENCHMARK SUITE (Python 3.10 - 3.14 compatible, Free-Threading Ready)           |     |
|   +-----------------------------------------------------------------------------------------------+     |
|         |                                                                                               |
|         +---> Base Classifier Modules (`cfmnn/`): CFMNN, Simpson (1992), GFMM, Chunked Streaming        |
|         +---> Explainer Engines (`explainers/`): Recurrence SHAP, PACE-flip, Counterfactual, Decision  |
|         +---> Repair & Safety Gates (`repair/`): Self-Repair, Active Learning, Gate Caching, Prop 4   |
|         +---> Trust & Lie Detector (`trust/`): PD, EI, BoxEv, U*, PRS, Logistic Meta-Classifier        |
|         +---> Standardized Baselines (`baselines/`): EFMNN, KNEFMNN, RFMNN, IOL-GFMM, EBM, RF, 10-NN    |
|         |                                                                                               |
|   +-----------------------------------------------------------------------------------------------+     |
|   | NUMERICAL & SCIENTIFIC BACKEND (NumPy 1.26.4, SciPy 1.12.0, Scikit-Learn 1.4.1)               |     |
|   +-----------------------------------------------------------------------------------------------+     |
|         |                                                                                               |
|   +-----------------------------------------------------------------------------------------------+     |
|   | HARDWARE ACCELERATION: SIMD AVX-512 / AVX2 Vectorized Instruction Pipeline                     |     |
|   | Zero GPU / CUDA requirements; runs seamlessly on Linux, Windows, macOS, and Cloud Micro-VMs   |     |
|   +-----------------------------------------------------------------------------------------------+     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Hardware, Software Environment & Dependency Manifest

### Platform Compatibility
* **Python Runtime:** Python 3.10.x, 3.11.x, 3.12.x, 3.13.x, and verified on Python 3.14.0a3 (pre-release compatibility with free-threaded CPython).
* **Instruction Sets:** AVX-512, AVX2, FMA3, and ARM NEON (via vectorized NumPy/OpenBLAS loops).
* **Operating Systems:** Ubuntu Linux 20.04/22.04 LTS, Debian 12 (Bookworm), macOS Sonoma/Ventura (Apple Silicon M1/M2/M3 native), Microsoft Windows 11 Enterprise (PowerShell / Command Prompt).

### Pinned Dependency Lock (`requirements-lock.txt`)

```
# Core Numeric and Computational Backend
numpy==1.26.4
scipy==1.12.0
scikit-learn==1.4.1.post1

# Explainable AI & Counterfactual Baselines
interpret==0.5.1          # Microsoft Explainable Boosting Machines (EBM / GA2M)
dice-ml==0.11             # Microsoft Diverse Counterfactual Explanations (DiCE)
xgboost==2.0.3            # Extreme Gradient Boosting baseline

# Testing, Verification & Tooling
pytest==8.0.2
pytest-cov==4.1.0
joblib==1.3.2
tabulate==0.9.0
```

### Containerized Reproducibility (`Dockerfile`)

```dockerfile
FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*
COPY requirements-lock.txt .
RUN pip install --upgrade pip && pip install -r requirements-lock.txt
COPY . .
CMD ["python", "-B", "-m", "unittest", "discover", "-s", "tests"]
```

---

## 3. Mathematical Foundations: Exact Formulas Used Across Modules

Every transformation in the codebase is derived from rigorous closed-form mathematical equations:

### 1. Deterministic Min-Max Preprocessing (Strict $[0, 1]^d$)
$$x_{i, k} = \frac{x_{\text{raw}, i, k} - \min_j X_{\text{train}, j, k}}{\max_j X_{\text{train}, j, k} - \min_j X_{\text{train}, j, k} + \epsilon}$$
*Rule:* Minimum and maximum vectors are fitted strictly on the training partition to prevent any data leakage.

### 2. Box Penalty Function ($f$)
$$f(x_k, V_{jk}, W_{jk}, \gamma) = \begin{cases}
0 & \text{if } V_{jk} \le x_k \le W_{jk} \\
\min(1.0, \gamma (V_{jk} - x_k)) & \text{if } x_k < V_{jk} \\
\min(1.0, \gamma (x_k - W_{jk})) & \text{if } x_k > W_{jk}
\end{cases}$$
Default parameter: $\gamma = 4.0$ (empirically validated as the optimal decay ramp).

### 3. Box Membership ($\mu_j$) & Class Aggregation ($M_c$)
$$\mu_j(\mathbf{x}) = 1.0 - \frac{1}{d} \sum_{k=1}^d f(x_k, V_{jk}, W_{jk}, \gamma) \in [0, 1]$$
$$M_c(\mathbf{x}) = \max_{j: \text{class}(B_j) = c} \mu_j(\mathbf{x})$$

### 4. Centroid Tie-Breaking Rule
When $|\arg\max_c M_c(\mathbf{x})| \ge 2$, the prediction is:
$$\hat{y}(\mathbf{x}) = \arg\min_{c \in \mathcal{T}(\mathbf{x})} \|\mathbf{x} - \mathbf{c}_c\|_2^2 = \arg\min_{c \in \mathcal{T}(\mathbf{x})} \sum_{k=1}^d (x_k - c_{ck})^2$$

### 5. Exact Shapley Coalition Value ($v(S)$)
$$v(S) = \frac{1}{|\mathcal{D}_{\text{bg}}|} \sum_{\mathbf{z} \in \mathcal{D}_{\text{bg}}} M_c(\mathbf{x}_S \cup \mathbf{z}_{\bar{S}})$$

### 6. Gaussian Kernel Weights for PACE-SHAP-cond ($w_i(S, \sigma)$)
$$w_i(S, \sigma) = \frac{\exp\left( - \frac{\|\mathbf{x}_S - \mathbf{z}_{i, S}\|_2^2}{2 \sigma^2} \right)}{\sum_{j=1}^N \exp\left( - \frac{\|\mathbf{x}_S - \mathbf{z}_{j, S}\|_2^2}{2 \sigma^2} \right)}$$

### 7. Explanation Instability ($\text{EI}$)
$$\text{EI}(\mathbf{x}) = \frac{1}{K} \sum_{k=1}^K \left( 1 - \frac{\boldsymbol{\phi}(\mathbf{x}) \cdot \boldsymbol{\phi}(\mathbf{x}^{(k)})}{\|\boldsymbol{\phi}(\mathbf{x})\|_2 \|\boldsymbol{\phi}(\mathbf{x}^{(k)})\|_2 + \epsilon} \right)$$

---

## 4. Codebase Architecture: Module-by-Module Source Code Mapping

The codebase is organized into five functional packages:

```
pace-cfmnn-main/
│
├── cfmnn/                               # BASE CLASSIFIER IMPLEMENTATION
│   ├── cfmnn.py                         # Class: CFMNN (fit, predict, predict_proba, expand_box)
│   ├── simpson_fmnn.py                  # Class: SimpsonFMNN (original 1992 contraction logic)
│   ├── gfmm.py                          # Class: GFMM (min-operator fuzzy network)
│   └── covertype_chunked.py             # Function: train_covertype_chunked (cache-blocked streaming)
│
├── explainers/                          # EXPLAINABILITY & RECOURSE SUITE
│   ├── shap_exact.py                    # Function: compute_exact_shap_recurrence (O(K * 2^d))
│   ├── shap_antithetic.py               # Function: compute_antithetic_shap (32 paired permutations)
│   ├── shap_conditional.py              # Function: compute_conditional_shap (Gaussian bandwidth sigma)
│   ├── flip.py                          # Function: compute_flip_quantiles (bounds-pruned recourse)
│   ├── flip_exact.py                    # Function: solve_exact_linear_flip (Proposition 8 solver)
│   ├── decision.py                      # Function: explain_tied_decision (Proposition 10 centroid SHAP)
│   ├── counterfactual.py                # Function: find_pace_sparse_counterfactual (locked features)
│   └── box_native.py                    # Class: BoxNativeExplainer (decomp, contrastive, multi)
│
├── repair/                              # SURGICAL REPAIR & CERTIFICATION
│   ├── self_repair.py                   # Function: execute_self_repair (zero-label shrink/expand)
│   ├── active_learning.py               # Class: ActiveLearner (margin, uncertainty, disagreement)
│   ├── validation_gate.py               # Class: ValidationGate (delta Macro-F1 and recall circuit breaker)
│   ├── certificate.py                   # Function: compute_proposition4_certificate (spatial bounds)
│   └── cache_gate.py                    # Class: CachedValidationGate (sub-2 ms evaluation cache)
│
├── trust/                               # TRUST SIGNALS & ERROR DETECTION
│   ├── signals.py                       # Functions: compute_pd, compute_ei, compute_box_evidence
│   ├── pace_score.py                    # Function: compute_prs (non-parametric rank aggregator)
│   └── meta_classifier.py               # Class: ErrorMetaClassifier (regularized logistic combiner)
│
└── baselines/                           # EXTERNAL & FUZZY COMPARISON BASELINES
    ├── efmnn.py                         # Enhanced Fuzzy Min-Max
    ├── knefmnn.py                       # K-Nearest Enhanced Fuzzy Min-Max
    ├── rfmnn.py                         # Robust Fuzzy Min-Max
    └── iol_gfmm.py                      # Improved Online Learning GFMM
```

---

## 5. Algorithmic Pseudocode: The Core CFMNN Single-Pass Trainer

```python
class CFMNN:
    def __init__(self, theta=0.20, gamma=4.0, centroid_rule="fixed", expansion="search"):
        self.theta = theta
        self.gamma = gamma
        self.centroid_rule = centroid_rule
        self.expansion = expansion
        self.boxes = [] # List of HyperBox objects
        self.centroids = {} # Class -> np.ndarray
        
    def fit(self, X, y):
        # 1. Precompute fixed empirical class centroids
        classes = np.unique(y)
        for c in classes:
            self.centroids[c] = np.mean(X[y == c], axis=0)
            
        # 2. Single-pass online expansion loop
        for i in range(len(X)):
            x_i, y_i = X[i], y[i]
            candidate_boxes = [b for b in self.boxes if b.class_id == y_i]
            
            # Find eligible boxes satisfying maximum side length theta
            eligible = []
            for b in candidate_boxes:
                V_new = np.minimum(b.V, x_i)
                W_new = np.maximum(b.W, x_i)
                if np.max(W_new - V_new) <= self.theta:
                    eligible.append((b, V_new, W_new))
                    
            if eligible:
                # Expand best candidate box (minimal volume increase)
                best_box, V_best, W_best = min(
                    eligible, 
                    key=lambda item: np.sum((item[2] - item[1]) - (item[0].W - item[0].V))
                )
                best_box.V = V_best
                best_box.W = W_best
                best_box.member_count += 1
            else:
                # Instantiate new micro-box at point x_i
                new_box = HyperBox(V=np.copy(x_i), W=np.copy(x_i), class_id=y_i)
                self.boxes.append(new_box)
        return self
```
