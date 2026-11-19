# 04. End-to-End Workflow and System Architecture

## 1. Architectural System Overview

The PACE-CFMNN framework is an integrated, closed-loop machine learning system that unites:
1. **Geometric Representation Learning:** Training a Centroid-Based Fuzzy Min-Max Neural Network via single-pass hyperbox expansion in $[0, 1]^d$.
2. **Deterministic Closed-Form Explainability (PACE):** Computing exact Shapley values (`PaceShap`), exact single-feature decision flips (`PaceFlip`), and sparse counterfactuals (`sparse_counterfactual`) directly from model parameters without black-box sampling.
3. **Intrinsic Safety & Trust Telemetry:** Evaluating label-free prediction disagreement (PD), explanation instability (EI), and corrected tie uncertainty ($U^*$).
4. **Audited Self-Repair & Active Refinement:** Applying surgical hyperbox boundary cuts to misclassified points behind a statistical validation gate.
5. **Provable Invariance Certification:** Generating formal mathematical certificates that guarantee zero prediction flips outside a bounded spatial impact set.

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                        PACE-CFMNN END-TO-END PIPELINE ARCHITECTURE                                      |
|                                                                                                                         |
|       [ Raw Input Data ]                                                                                                |
|               |                                                                                                         |
|               v                                                                                                         |
|       [ Min-Max Normalizer ]  X_scaled in [0, 1]^(N x d)                                                                |
|               |                                                                                                         |
|               v                                                                                                         |
|       +-----------------------------------------------------------------------+                                         |
|       |                     PHASE 1: CFMNN TRAINING                           |                                         |
|       |  Single-Pass Incremental Fit (Algorithm 2)                            |                                         |
|       |  - Bounded Expansion: sum_i (w_i - v_i) <= d * theta                  |                                         |
|       |  - Search Mode: Best same-class box that can expand                   |                                         |
|       |  - Fixed Centroid Tracking: mu_j = (1 / N_j) * sum x_r                |                                         |
|       +-----------------------------------------------------------------------+                                         |
|               |                                                                                                         |
|               +-------------------------------+-------------------------------+                                         |
|               |                               |                               |                                         |
|               v                               v                               v                                         |
|       +-----------------------+       +-----------------------+       +-----------------------+                         |
|       |  PHASE 2: INFERENCE   |       |  PHASE 3: EXPLAIN     |       |  PHASE 4: TRUST & QA  |                         |
|       |  - Ramp Membership    |       |  - PACE-SHAP (Exact)  |       |  - Perturbation PD    |                         |
|       |  - Tie-Tolerance tau  |       |  - PACE-flip (Bounds) |       |  - Explanation EI     |                         |
|       |  - Nearest Centroid   |       |  - PACE-sparse (CF)   |       |  - Corrected U*       |                         |
|       +-----------------------+       +-----------------------+       +-----------------------+                         |
|               |                               |                               |                                         |
|               |                               |                               |                                         |
|               +-------------------------------+-------------------------------+                                         |
|                                               |                                                                         |
|                                               v                                                                         |
|                               +-------------------------------+                                                         |
|                               |  Error Detection / Audit Flag |                                                         |
|                               |  (Training Error / High PRS)  |                                                         |
|                               +-------------------------------+                                                         |
|                                               |                                                                         |
|                                               v                                                                         |
|       +-----------------------------------------------------------------------+                                         |
|       |                 PHASE 5: SURGICAL GATED SELF-REPAIR                   |                                         |
|       |  1. Identify Overlapping Winning Wrong Box k (Proposition 1)          |                                         |
|       |  2. Execute Infinitesimal Face Cut on Feature i                       |                                         |
|       |  3. Recompute Box Member Counts & Centroids                           |                                         |
|       |  4. Query Gate (Delta Macro-F1 <= 0.5%, Delta Recall <= 1.0%)         |                                         |
|       |       |                                                               |                                         |
|       |       +---> Reject: Rollback in-place state mutation                  |                                         |
|       |       +---> Accept: Commit new box geometry                           |                                         |
|       +-----------------------------------------------------------------------+                                         |
|                                               |                                                                         |
|                                               v                                                                         |
|       +-----------------------------------------------------------------------+                                         |
|       |                 PHASE 6: FORMAL INVARIANCE CERTIFICATION              |                                         |
|       |  Proposition 4: Compare Top Groups T_before(x) & T_after(x)           |                                         |
|       |  Impact Certificate: C_cert = {x : Changed Box in T(x)}               |                                         |
|       |  Guarantee: Zero prediction changes on all x notin C_cert             |                                         |
|       +-----------------------------------------------------------------------+                                         |
+-------------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Step-by-Step System Execution Flow

### Stage 1: Data Ingestion & Preprocessing
* **Input:** Raw multidimensional tabular matrix $X_{\text{raw}} \in \mathbb{R}^{N \times d}$, label vector $y \in \{0, \dots, C-1\}^N$.
* **Transformation:** Feature-wise min-max affine projection fitted strictly on training rows:
  $$X_{nj} = \frac{X_{\text{raw}, nj} - \min_r X_{\text{raw}, rj}}{\max_r X_{\text{raw}, rj} - \min_r X_{\text{raw}, rj} + \epsilon} \in [0, 1]$$
* **Output:** Normalized matrix $X \in [0, 1]^{N \times d}$.

### Stage 2: Single-Pass CFMNN Fit
* **Execution:** `CFMNN.fit(X_train, y_train)` in `pace_cfmnn/cfmnn_core.py`.
* **State Variables Maintained:**
  - `V_`: Lower box corner matrix, shape $(H, d)$.
  - `W_`: Upper box corner matrix, shape $(H, d)$, with $V_{ji} \le W_{ji}$.
  - `centroid_`: Centroid coordinates matrix, shape $(H, d)$, where $\mu_j = \frac{1}{N_j} \sum_{r \in \text{members}(j)} x_r$.
  - `box_class_`: Class index mapping, shape $(H,)$, where $c_j \in \{0, \dots, C-1\}$.
  - `members_`: List of length $H$, containing arrays of training point indices absorbed by each box.
  - `count_`: Point count vector, shape $(H,)$.
  - `sum_`: Vector sum of member points, shape $(H, d)$.

### Stage 3: Inference & Prediction Pipeline
* **Execution:** `CFMNN.predict_details(X_query)` for query batch $X_{\text{query}} \in [0, 1]^{n \times d}$.
* **Mathematical Operations:**
  1. Evaluate 1-D feature terms for all query points against all $H$ boxes:
     $$m_{j, i}(x_i) = \frac{1}{2} \Big[ f(x_i - W_{ji}) + f(V_{ji} - x_i) \Big] \in [0.5, 1.0]$$
     Tensor shape: $(n, H, d)$.
  2. Compute average box membership across features:
     $$M_{nj} = b_j(x_n) = \frac{1}{d} \sum_{i=1}^d m_{j, i}(x_{n, i})$$
     Tensor shape: $(n, H)$.
  3. Identify maximum membership:
     $$b^*_n = \max_{j \in \{1, \dots, H\}} M_{nj}$$
  4. Form tied top group within tolerance $\tau = 10^{-9}$:
     $$T(x_n) = \{ j : M_{nj} \ge b^*_n - \tau \}$$
  5. Compute squared Euclidean distances to centroids:
     $$D_{nj} = \|x_n - \mu_j\|_2^2 = \sum_{i=1}^d (x_{ni} - \mu_{ji})^2$$
  6. Break ties by finding the nearest centroid:
     $$j^*_n = \arg\min_{j \in T(x_n)} D_{nj}$$
     Remaining ties resolve to the lowest box index $j$.
* **Return Structure:** `Prediction` dataclass containing predicted class $y_{\text{pred}}$, winning box $j^*$, class supports $s_c(x)$, and runner-up class index.

---

### Stage 4: PACE Feature Attribution & Explainability

```
+----------------------------------------------------------------------------------------------------+
|                                    PACE-SHAP EXECUTION WORKFLOW                                    |
|                                                                                                    |
|   Query Point x in R^d                 Background Rows Z in R^(R x d)                              |
|           |                                       |                                                |
|           +-------------------+-------------------+                                                |
|                               |                                                                    |
|                               v                                                                    |
|           Delta Matrix Evaluation (Proposition 2):                                                 |
|           Delta_(r, j, i) = [ m_(j, i)(x_i) - m_(j, i)(z_(r, i)) ] / d                             |
|           Tensor Shape: (R, H, d)                                                                  |
|                               |                                                                    |
|                               v                                                                    |
|           Exact Floor & Ceiling Bounding Box Pruning (Proposition 3):                              |
|           Floor_j = b_j(z_r) + sum_i min(0, Delta_(r, j, i))                                       |
|           Ceil_j  = b_j(z_r) + sum_i max(0, Delta_(r, j, i))                                       |
|           Discard box j if Ceil_j < max_(k in class) Floor_k  (71% - 98% skipped!)                |
|                               |                                                                    |
|                               v                                                                    |
|           +---------------------------------------+---------------------------------------+        |
|           | If d <= 12 (Exact Enumeration)        | If d > 12 (Antithetic Permutation)    |        |
|           | Evaluate all 2^d coalitional margins  | Sample 32 balanced antithetic orders  |        |
|           | via Dynamic Recurrence Engine         | Shared random streams across instances|        |
|           | Exact Shapley Formula (Weight W_S)    | Incremental O(H) feature additions    |        |
|           +---------------------------------------+---------------------------------------+        |
|                               |                                                                    |
|                               v                                                                    |
|           Exact Interventional Attribution Vector: phi in R^d                                      |
|           Guaranteed Stability: 1.000000000000                                                     |
+----------------------------------------------------------------------------------------------------+
```

---

### Stage 5: Surgical Gated Self-Repair (SR)

When a model misclassifies a training point $x_r$ ($y_{\text{pred}}(x_r) \ne y_{\text{true}}(x_r)$):

1. **Proposition 1 Invocation:** Verify that $x_r$ lies inside winning box $k$ with $c_k \ne y_{\text{true}}(x_r)$ and membership $1.0$.
2. **Face Selection:** Choose feature $i$ causing the overlap. In standard SR (`"volume"`), select the dimension where the cut removes the smallest volume fraction:
   $$i^* = \arg\min_i \text{share\_cut}(k, x_r, i)$$
3. **Boundary Translation:** Execute `face_cut()`:
   - If $x_{r, i} - V_{ki} \le W_{ki} - x_{r, i}$, move lower face: $V_{ki} \leftarrow x_{r, i} + 10^{-3}$.
   - Else, move upper face: $W_{ki} \leftarrow x_{r, i} - 10^{-3}$.
4. **Member Re-assignment & Centroid Re-centering:**
   - Filter training points $\text{members}(k)$ to retain only rows falling within updated bounds.
   - If $\text{members}(k)$ becomes empty, delete box $k$.
   - Recompute $\mu_k = \frac{1}{|\text{members}(k)|} \sum_{p \in \text{members}(k)} x_p$.
5. **Statistical Acceptance Gate Check:**
   - Measure macro-F1 and per-class recall across acceptance rows.
   - If $\Delta \text{Macro-F1} > -0.5\%$ AND $\Delta \text{Recall}_c > -1.0\%$ for all classes $c$, and $H \le \text{box\_cap}$: **ACCEPT**.
   - Otherwise: **ROLLBACK** (restore mutated box dimensions and member sets in place).

---

### Stage 6: Formal Invariance Certification (Proposition 4)

After an accepted box edit ($M \to M'$):
1. Identify mutated boxes: $C_{\text{changed}} = \{k : V_k \ne V'_k \lor W_k \ne W'_k \lor \mu_k \ne \mu'_k\}$.
2. Query tied top groups before edit $T(x)$ and after edit $T'(x)$ across all test points $x \in X_{\text{test}}$.
3. Compute impact certificate:
   $$C_{\text{cert}}(x) = \mathbb{I}\Big( (T(x) \cap C_{\text{changed}} \ne \emptyset) \lor (T'(x) \cap C_{\text{changed}} \ne \emptyset) \Big)$$
4. **Mathematical Guarantee:**
   $$\forall x \notin C_{\text{cert}}, \quad y_{\text{pred}}(x; M') \equiv y_{\text{pred}}(x; M)$$
   Zero predictions flip outside $C_{\text{cert}}$.

---

## 3. Data Structures & Tensor Dimensionality Ledger

To ensure technical precision for systems engineering and reimplementation, the table below documents every core dataclass, attribute, and tensor shape across the entire lifecycle:

| Class / Structure | Field Name | Data Type | Exact Tensor Shape | Physical / Semantic Description |
|---|---|---|---|---|
| **`CFMNN`** | `V_` | `np.float64` | $(H, d)$ | Lower corners of all hyperboxes in $[0, 1]^d$ |
| | `W_` | `np.float64` | $(H, d)$ | Upper corners of all hyperboxes in $[0, 1]^d$ ($V \le W$) |
| | `centroid_` | `np.float64` | $(H, d)$ | Arithmetic mean coordinates $\mu_j$ of absorbed training points |
| | `box_class_` | `np.int64` | $(H,)$ | Categorical target class index $c_j \in \{0, \dots, C-1\}$ |
| | `count_` | `np.int64` | $(H,)$ | Total number of training instances $N_j$ absorbed by box $j$ |
| | `members_` | `list[np.ndarray]` | Length $H$, rows $(N_j,)$ | Integer row indices of training points inside box $j$ |
| **`Prediction`** | `y_pred` | `np.ndarray` | $(n,)$ | Predicted class labels in original data domain |
| | `cls` | `np.int64` | $(n,)$ | Predicted class indices $A \in \{0, \dots, C-1\}$ |
| | `winner` | `np.int64` | $(n,)$ | Index $j^* \in \{0, \dots, H-1\}$ of winning hyperbox |
| | `n_tied` | `np.int64` | $(n,)$ | Cardinality $|T(x)|$ of boxes tied at top membership |
| | `support` | `np.float64` | $(n, C)$ | Raw max class membership $s_c(x) = \max_{j \in c} b_j(x)$ |
| | `support_rescaled`| `np.float64` | $(n, C)$ | Affine rescaled support $s'_c(x) = 2 s_c(x) - 1 \in [0, 1]$ |
| | `runner_up` | `np.int64` | $(n,)$ | Index $B$ of competitor class with second-highest support |
| **`Decomposition`** | `phi` | `np.float64` | $(n, d)$ | Additive feature attributions for winning box membership |
| | `base` | `np.float64` | $(n,)$ | Expected membership $\mathbb{E}_{z \sim B}[b_{j^*}(z)]$ |
| **`Contrastive`** | `e` | `np.float64` | $(n, d)$ | Feature difference vector explaining "Why class $A$, not class $B$" |
| | `mode` | `np.str_` | $(n,)$ | Explanation mode: `"membership"`, `"tie"`, or `"none"` |
| **`PaceShap`** | `phi` | `np.float64` | $(n, d)$ | Exact or permutation Shapley values of margin $g_A(x)$ |
| | `fx` | `np.float64` | $(n,)$ | Prediction margin $g_A(x)$ evaluated at point $x$ |
| | `f0` | `np.float64` | $(n,)$ | Baseline margin $\frac{1}{R} \sum_{r=1}^R g_A(z_r)$ |
| **`Signals`** | `pd` | `np.float64` | $(n,)$ | Prediction disagreement fraction across $K=50$ neighbors |
| | `ei` | `np.float64` | $(n,)$ | Explanation instability (cosine distance) across neighbors |
| | `u_star` | `np.float64` | $(n,)$ | Corrected uncertainty with tie priority in $[0.5, 1.0]$ |
| **`Gate`** | `max_f1_drop` | `float` | Scalar ($0.5$) | Maximum allowable macro-F1 percentage drop on gate rows |
| | `max_recall_drop`| `float` | Scalar ($1.0$) | Maximum allowable recall percentage drop for any single class |
| **`Certificate`** | `strict` | `np.bool_` | $(n,)$ | Tight impact certificate boolean mask ($C_{\text{cert}}$) |

---

## 4. Cross-References to Detailed Subsystems

- For exact mathematical derivations of $m_{ji}(x)$ and Proposition proofs: [`06_Mathematical_Foundations_and_Theoretical_Formulation.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/06_Mathematical_Foundations_and_Theoretical_Formulation.md)
- For the architectural breakdown of each individual subsystem block: [`07_Component_By_Component_Architectural_Design.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/07_Component_By_Component_Architectural_Design.md)
- For the code walkthrough of `cfmnn_core.py` and `explain.py`: [`08_Codebase_Implementation_Deep_Dive_Core_Modules.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/08_Codebase_Implementation_Deep_Dive_Core_Modules.md)
- For the code walkthrough of `refine.py`, `active.py`, and `audit.py`: [`09_Codebase_Implementation_Deep_Dive_Training_and_Utils.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/09_Codebase_Implementation_Deep_Dive_Training_and_Utils.md)
- For granular timing breakdowns across forward passes, explanations, and repair: [`15_Component_Wise_Training_Time_and_Convergence_Analysis.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/15_Component_Wise_Training_Time_and_Convergence_Analysis.md) and [`16_Component_Wise_Testing_Time_Latency_and_Compute_Profile.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/16_Component_Wise_Testing_Time_Latency_and_Compute_Profile.md)
- For deep analysis of self-repair mechanics and certificate bounds: [`22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md)
