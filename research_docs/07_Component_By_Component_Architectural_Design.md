# 07. Component-By-Component Architectural Design

## 1. System Module Hierarchy

The PACE-CFMNN architecture is partitioned into ten specialized, highly decoupled subsystems that communicate via structured NumPy arrays and immutable dataclasses:

```
+----------------------------------------------------------------------------------------------------+
|                                    PACE-CFMNN COMPONENT TOPOLOGY                                   |
|                                                                                                    |
|    +-----------------------------+               +-----------------------------+                   |
|    |  1. CFMNN Core Classifier   |<------------->|  2. PACE-SHAP Explainer     |                   |
|    |  (Box Geometry & Centroids) |               |  (Additive Margin Shapley)  |                   |
|    +-----------------------------+               +-----------------------------+                   |
|                   ^                                             ^                                  |
|                   |                                             |                                  |
|                   v                                             v                                  |
|    +-----------------------------+               +-----------------------------+                   |
|    |  3. PACE-flip Explainer     |               |  4. PACE-Decision Explainer |                   |
|    |  (Pruned Occlusion Bounds)  |               |  (Tie-Aware Categorical)    |                   |
|    +-----------------------------+               +-----------------------------+                   |
|                   ^                                             ^                                  |
|                   |                                             |                                  |
|                   v                                             v                                  |
|    +-----------------------------+               +-----------------------------+                   |
|    |  5. Contrastive Engine      |               |  6. Counterfactual Engine   |                   |
|    |  (Why Class A, Not B?)      |               |  (Box-Directed & Sparse)    |                   |
|    +-----------------------------+               +-----------------------------+                   |
|                   ^                                             ^                                  |
|                   |                                             |                                  |
|                   v                                             v                                  |
|    +-----------------------------+               +-----------------------------+                   |
|    |  7. Trust Signals & PRS     |<------------->|  8. Trust Combiner Engine   |                   |
|    |  (PD, EI, U*, Box Evidence) |               |  (Calibrated Meta-Learner)  |                   |
|    +-----------------------------+               +-----------------------------+                   |
|                   ^                                             ^                                  |
|                   |                                             |                                  |
|                   v                                             v                                  |
|    +-----------------------------+               +-----------------------------+                   |
|    |  9. Surgical Self-Repair    |<------------->| 10. Audit & Certification   |                   |
|    |  (Gated Boundary Shrinks)   |               |  (Invariance Certificates)  |                   |
|    +-----------------------------+               +-----------------------------+                   |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Granular Component Breakdowns

### Component 1: `CFMNN` Core Classifier (`pace_cfmnn/cfmnn_core.py`)
* **Role & Purpose:** The foundational geometric model. Represents data partitions using hyperboxes and internal centroids. Implements single-pass training without backpropagation.
* **Input Interface:**
  - `X`: Scaled feature array, shape $(N, d)$, dtype `float64` in $[0, 1]$.
  - `y`: Categorical target vector, shape $(N,)$, dtype `int64` in $\{0, \dots, C-1\}$.
* **Internal Transformations:**
  - Maintains `_ClassBuffer` dynamically during fit.
  - Checks hyperbox budget condition: $\sum_{i=1}^d (w_{ji} - v_{ji}) \le d \cdot \theta$.
  - Executes `"search"` expansion: scans existing same-class boxes in descending membership order to find the first candidate that can absorb the point without exceeding $\theta$; if none can expand, instantiates a new point box ($v = w = x$).
  - Accumulates `sum_` and `count_` for every accepted point (`centroid_rule="fixed"`).
* **Output Interface:**
  - Returns a populated `CFMNN` instance with fitted arrays: `V_` $(H, d)$, `W_` $(H, d)$, `centroid_` $(H, d)$, `box_class_` $(H,)$, `members_` (list of member indices).
* **Downstream Connection:** Serves as the structural substrate for all explainers, trust monitors, and repair operators.

---

### Component 2: `PaceShap` Explainer (`pace_cfmnn/explain.py`)
* **Role & Purpose:** Computes exact or antithetic permutation Shapley values of the prediction margin $g_A(x)$ without querying the classifier forward pass.
* **Input Interface:**
  - `model`: Fitted `CFMNN` instance.
  - `background`: Reference dataset $Z$, shape $(R, d)$ (default $R = 200$).
  - `exact_max_d`: Dimensionality threshold for exact evaluation (default $12$).
  - `sigma`: Optional Gaussian bandwidth for empirical conditional game (default `None` for interventional).
* **Internal Transformations:**
  - Evaluates additive matrix $\Delta_{r, j, i} = \frac{1}{d}[m_{ji}(x_i) - m_{ji}(z_{r, i})]$.
  - Evaluates Proposition 3 pruning: calculates $\text{floor}_j$ and $\text{ceil}_j$ across background rows, discarding boxes that cannot achieve class maximum.
  - If $d \le 12$: Evaluates all $2^d$ subset margins via dynamic recurrence (`exact_algo="recurrence"`: $\mathcal{O}(K \cdot 2^d)$ additions) or matrix multiplication (`exact_algo="matmul"`).
  - If $d > 12$: Samples 32 antithetic permutations per background row using shared random seeds.
  - If `sigma` is provided: Applies conditional Gaussian weights $w_z(S) = \exp\big(-\frac{D_S(x, z)^2}{2\sigma^2}\big)$ based on normalized subset feature distance.
* **Output Interface:**
  - Attribution matrix $\phi \in \mathbb{R}^{n \times d}$, where $\sum_{i=1}^d \phi_{ni} = g_A(x_n) - \mathbb{E}_z[g_A(z)]$.
* **Downstream Connection:** Provides feature importance rankings for faithfulness benchmarks, error diagnostics, and feature-guided repair.

---

### Component 3: `PaceFlip` Explainer (`pace_cfmnn/explain.py`)
* **Role & Purpose:** Determines which single-feature changes cause the model's categorical prediction to flip, and calculates the resulting margin reduction.
* **Input Interface:**
  - `model`: Fitted `CFMNN`.
  - `background`: Background dataset $Z \in \mathbb{R}^{R \times d}$.
  - `n_quantiles`: Number of mid-quantile replacement values per feature (default 32).
* **Internal Transformations:**
  - Extracts 32 mid-quantiles $Q \in \mathbb{R}^{d \times 32}$ from background data.
  - Evaluates replacement terms $m_{ji}(Q_{ik})$.
  - Applies analytical bounds pruning: skips boxes whose ceiling falls below competitor class floors (eliminating $71\% - 98\%$ of boxes).
  - Evaluates class winner and nearest-centroid tie-breaker in closed form across all $(n, d, 32)$ replacement states.
* **Output Interface:**
  - `flip`: Probability matrix in $[0, 1]^{n \times d}$ (fraction of 32 replacements that change the predicted class).
  - `drop`: Expected margin drop matrix in $\mathbb{R}^{n \times d}$.
* **Downstream Connection:** Provides ultra-fast ($< 1$ ms) feature sensitivity scores for real-time monitoring and active sample prioritization.

---

### Component 4: `PaceDecision` Explainer (`pace_cfmnn/decision.py`)
* **Role & Purpose:** Implements Proposition 10 to compute exact Shapley values directly over the hard categorical decision indicator $\mathbb{I}(\hat{y}(x) = A)$, fully accounting for centroid tie-breaking on tied points.
* **Input Interface:**
  - `model`: Fitted `CFMNN`.
  - `background`: Reference rows $Z \in \mathbb{R}^{R \times d}$.
* **Internal Transformations:**
  - For each feature subset $S$, evaluates the complete inference pipeline including centroid distance resolution $D_j(h_S)$ on tied subsets.
* **Output Interface:**
  - Decision Shapley matrix $\phi^{\text{dec}} \in \mathbb{R}^{n \times d}$.
* **Downstream Connection:** Explains boundary instances where margin is zero ($g_A(x) = 0$) but centroid distance decides the winning class.

---

### Component 5: `Contrastive` & `Decomposition` Engines (`pace_cfmnn/explain.py`)
* **Role & Purpose:**
  - `contrastive()`: Answers "Why class $A$ and not runner-up class $B$?"
  - `decomposition()`: Breaks down winning box membership into centered feature contributions.
* **Internal Transformations:**
  - `contrastive()` operates in two modes:
    - *Membership Mode:* $e_i = \frac{1}{d}(m_{Ai} - m_{Bi})$, where $\sum e_i = b_A(x) - b_B(x)$.
    - *Tie Mode:* When $b_A$ and $b_B$ tie within $\tau$, $e_i = (x_i - \mu_{Bi})^2 - (x_i - \mu_{Ai})^2$, where $\sum e_i = \|x - \mu_B\|^2 - \|x - \mu_A\|^2$.
* **Output Interface:**
  - `Contrastive` dataclass with per-feature vector $e \in \mathbb{R}^{n \times d}$ and mode indicators.
* **Downstream Connection:** Directly guides explanation instability (EI) tracking and explanation-guided box shrinking.

---

### Component 6: Counterfactual Engine (`pace_cfmnn/counterfactual.py`)
* **Role & Purpose:** Generates minimal, plausible counterfactual modifications to achieve a desired target class label $t \ne \hat{y}(x)$.
* **Input Interface:**
  - `x`: Query instance, shape $(d,)$.
  - `target`: Target class label $t$.
  - `locked`: Boolean mask of immutable features (e.g., age, sex).
* **Internal Transformations:**
  - `counterfactual` (PACE): Ranks target-class boxes by L1 distance on mutable features, clips $x$ onto candidate box boundaries $[V_k, W_k]$, steps toward the centroid if tied, and applies greedy sparsification.
  - `sparse_counterfactual` (PACE-sparse): Combinatorially searches feature subsets (exact up to 65,536 sets per size) to find the minimum number of feature shifts that flip the prediction.
* **Output Interface:**
  - `Counterfactual` dataclass: `found` (bool), `x_cf` (vector), `l1` (distance), `n_changed` (sparsity), `plausibility` (L1 distance to nearest target-class training instance).
* **Downstream Connection:** Generates actionable recourse for end-users and adversarial robustness testing.

---

### Component 7: Trust Signal Engine (`pace_cfmnn/signals.py`, `neighborhood.py`)
* **Role & Purpose:** Computes native, label-free reliability telemetry for deployed predictions.
* **Input Interface:**
  - `X`: Query batch, shape $(n, d)$.
* **Internal Transformations:**
  - Generates $K = 50$ neighbor perturbations per instance scaled to $5\%$ of feature empirical standard deviation.
  - Evaluates **Prediction Disagreement (PD):** Fraction of perturbations with flipped predicted class.
  - Evaluates **Explanation Instability (EI):** Mean cosine distance between contrastive explanation vectors across perturbations.
  - Evaluates **Corrected Uncertainty ($U^*$):** For tied points, evaluates normalized centroid ratio $\frac{1 + \|x-\mu_A\|/\|x-\mu_B\|}{2} \in [0.5, 1.0]$; for untied points, evaluates $\frac{1 - (s'_1 - s'_2)}{2} \in [0.0, 0.5]$.
  - Evaluates **Box Evidence:** Minimum distance from $x$ to training points absorbed by winning box $j^*$.
* **Output Interface:**
  - `Signals` dataclass containing arrays for `pd`, `ei`, `u_star`, and `box_evidence`.
* **Downstream Connection:** Feeds the PACE Refinement Score (PRS) and active learning query selectors.

---

### Component 8: Trust Combiner Engine (`pace_cfmnn/trust_combine.py`)
* **Role & Purpose:** Calibrates and combines multiple error detection signals using a meta-classifier fitted on labeled validation data.
* **Input Interface:**
  - Multimodal signal vectors (PD, $U^*$, Box Evidence, kNN disagreement, Trust Score, Order Ensemble disagreement).
* **Internal Transformations:**
  - Fits an $\ell_2$-regularized Logistic Regression classifier to predict $P(\hat{y}(x) \ne y(x))$.
* **Output Interface:**
  - Unified error probability score in $[0, 1]$.
* **Downstream Connection:** Achieves SOTA error detection AUPRC on complex benchmarks (RQ2 v2).

---

### Component 9: Surgical Self-Repair & Gate Engine (`pace_cfmnn/refine.py`, `active.py`)
* **Role & Purpose:** Modifies hyperbox geometry to eliminate misclassifications without retraining.
* **Input Interface:**
  - Misclassified point $(x_r, y_r)$, target box $k$, chosen feature dimension $i$.
* **Internal Transformations:**
  - Evaluates `face_cut()`: Moves boundary face $V_{ki}$ or $W_{ki}$ by $10^{-3}$ past $x_{ri}$.
  - Prunes member array $\text{members}(k)$ and updates $\mu_k$.
  - Executes `Gate.check()`: Validates that macro-F1 drop $\le 0.5\%$ and class recall drop $\le 1.0\%$ across acceptance instances.
  - If check fails: Restores mutated box bounds and member lists in place.
* **Output Interface:**
  - Returns `EditSummary`: status (`"accepted"` / `"rejected"`), feature cut, member delta.
* **Downstream Connection:** Executes closed-loop model maintenance.

---

### Component 10: Audit & Certification Engine (`pace_cfmnn/audit.py`)
* **Role & Purpose:** Implements Propositions 4 and 7 to provide mathematical non-interference certificates.
* **Input Interface:**
  - Model before edit $M$, model after edit $M'$, evaluation dataset $X_{\text{eval}}$.
* **Internal Transformations:**
  - Identifies mutated box indices: $\mathcal{C}_{\text{changed}}$.
  - Evaluates tied top groups $T_M(x)$ and $T_{M'}(x)$ across $X_{\text{eval}}$.
  - Forms tight certificate boolean mask:
    $$C_{\text{cert}}(x) = \mathbb{I}\Big( (T_M(x) \cap \mathcal{C}_{\text{changed}} \ne \emptyset) \lor (T_{M'}(x) \cap \mathcal{C}_{\text{changed}} \ne \emptyset) \Big) \land \text{multi\_class\_tied}$$
  - Computes empirical churn: $\frac{1}{N} \sum \mathbb{I}(\hat{y}(x; M) \ne \hat{y}(x; M'))$.
* **Output Interface:**
  - Boolean mask array `strict`, shape $(n,)$, and churn statistics dictionary.
* **Downstream Connection:** Provides mathematical verification and regulatory safety compliance.

---

## 3. Cross-References

- For the file-by-file code walkthrough of core modules: [`08_Codebase_Implementation_Deep_Dive_Core_Modules.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/08_Codebase_Implementation_Deep_Dive_Core_Modules.md)
- For the file-by-file code walkthrough of training and utility modules: [`09_Codebase_Implementation_Deep_Dive_Training_and_Utils.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/09_Codebase_Implementation_Deep_Dive_Training_and_Utils.md)
- For training dynamics and optimization strategy: [`10_Training_Pipeline_and_Optimization_Strategy.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/10_Training_Pipeline_and_Optimization_Strategy.md)
- For component-wise accuracy deltas and ablation studies: [`14_Component_Wise_Accuracy_and_Ablation_Study.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/14_Component_Wise_Accuracy_and_Ablation_Study.md)
- For granular testing latency and compute profiles per component: [`16_Component_Wise_Testing_Time_Latency_and_Compute_Profile.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/16_Component_Wise_Testing_Time_Latency_and_Compute_Profile.md)
