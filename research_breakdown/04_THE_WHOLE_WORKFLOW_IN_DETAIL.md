# Document 04: The Entire PACE-CFMNN Workflow in Detail

## 1. Executive Summary in Simple Human Terms

If you follow a single piece of data from the moment it enters the system to the moment it is classified, explained, checked for safety, and repaired, here is the complete journey:

1. **Step 1: Normalization:** The data is scaled strictly to numbers between $0.0$ and $1.0$. No future or test information is allowed to leak in.
2. **Step 2: Single-Pass Learning:** The model reads the training examples one by one. For each example, it looks to see if an existing box can absorb it without exceeding maximum box size $\theta$. If yes, the box stretches. If no, a new box is created.
3. **Step 3: Making a Prediction:** When a test sample arrives, all boxes calculate how close the sample is to them. The highest membership score wins. If boxes of two different classes tie with score $1.0$, the dispute is settled by measuring the physical distance to each class's true geometric average (the centroid).
4. **Step 4: Providing Multi-Angle Explanations:**
   - *Why this prediction?* -> Exact PACE-SHAP gives the exact mathematical credit score for each feature.
   - *What is the fastest way to flip the decision?* -> PACE-flip calculates the single smallest feature tweak in under 1 millisecond.
   - *How can I change the decision without changing my age?* -> PACE-Sparse provides a realistic, multi-feature recipe with locked features.
5. **Step 5: Safety Check (Error Lie Detector):** The system checks whether the prediction is sitting on a razor-thin boundary (Prediction Disagreement), whether the explanation is shaking under noise (Explanation Instability), and whether the point is far from real data (Box Evidence). If risk is high, it raises a red flag.
6. **Step 6: Fixing Mistakes (Self-Repair & Active Learning):** If an error occurs, the offending box is surgically shrunk or trimmed. Zero new labels are required!
7. **Step 7: Certified Safety Gate:** The Validation Gate checks that performance on holdout data didn't drop. Then it draws a spatial fence around the edit (Proposition 4 Certificate) mathematically guaranteeing that **0 predictions outside that fence were changed**.

```
+---------------------------------------------------------------------------------------------------------+
|                                  THE COMPLETE 7-STAGE PACE-CFMNN PIPELINE                               |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   STAGE 1: DATA INGESTION & CENTROIDS                                                                   |
|   Raw Data X_raw in R^{N x d}  --->  Scaled X in [0, 1]^d  --->  Compute Fixed Centroids c_c           |
|                                                                                                         |
|   STAGE 2: SINGLE-PASS ONLINE LEARNING                                                                  |
|   For each (x_i, y_i): Check Chebyshev Expansion Criterion: max(W' - V') <= theta                      |
|   Expand existing box OR instantiate new micro-box B_new (Zero backprop iterations!)                    |
|                                                                                                         |
|   STAGE 3: INFERENCE & CENTROID TIE-BREAKING                                                            |
|   Box Memberships mu_j(x)  --->  Class Max-Pool M_c(x)  --->  Tied? Settle via argmin ||x - c_c||_2   |
|                                                                                                         |
|   STAGE 4: FORENSIC AUDIT & EXPLANATIONS                                                                |
|   - Exact PACE-SHAP (Recurrence Lattice, r = 1.0000)                                                    |
|   - Sub-millisecond PACE-flip (32 Quantile Bounds Pruning)                                              |
|   - Actionable PACE-Sparse Recourse (Locked Immutable Features)                                         |
|                                                                                                         |
|   STAGE 5: TRUST SCORING & LIE DETECTOR                                                                |
|   - Intrinsic Signals: Disagreement (PD), Instability (EI), Evidence (BoxEv), Uncertainty (U*)         |
|   - Non-parametric PRS Aggregator & Logistic Error Meta-Classifier                                      |
|                                                                                                         |
|   STAGE 6: SURGICAL SELF-REPAIR (SR) & ACTIVE LEARNING (AL)                                             |
|   - Detect Overlapping Cross-Class Ties (Proposition 1)                                                 |
|   - Surgical Box Shrink / Micro-Delete / Micro-Expand (+1.5 to +6.0 Macro-F1 gain, 0 labels!)           |
|                                                                                                         |
|   STAGE 7: VALIDATION GATE & SPATIAL CERTIFICATION                                                      |
|   - Non-destructive Cache Gate Audit: Delta Macro-F1 >= -0.5%                                          |
|   - Compute Proposition 4 Certificate: Spatial Bounding Box C_{spatial}                                 |
|   - Mathematical Guarantee: 0 prediction leaks in the rest of the universe!                             |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Granular Walkthrough of the 7 Workflow Stages

### Stage 1: Data Ingestion & Deterministic Scaling
* **Input:** Raw dataset matrix $X_{\text{raw}} \in \mathbb{R}^{N \times d}$, categorical labels $y \in \{0, \dots, C-1\}^N$.
* **Leakage-Free Normalization:**
  $$\mathbf{x} = \text{clip}\left( \frac{\mathbf{x}_{\text{raw}} - \mathbf{x}_{\min}^{\text{train}}}{\mathbf{x}_{\max}^{\text{train}} - \mathbf{x}_{\min}^{\text{train}} + \epsilon}, \, 0.0, \, 1.0 \right)$$
* **Global Centroid Precomputation:** For each class $c$, the mean coordinate vector $\mathbf{c}_c = \frac{1}{|S_c|} \sum_{i \in S_c} \mathbf{x}_i$ is computed and stored permanently.

---

### Stage 2: Single-Pass Online Geometric Learning
* **Online Traversal:** Training instances $(\mathbf{x}_i, y_i)$ are processed strictly one by one.
* **Chebyshev Expansion Test:** For each candidate box $B_j$ belonging to class $y_i$:
  $$V'_{jk} = \min(V_{jk}, x_{ik}), \quad W'_{jk} = \max(W_{jk}, x_{ik})$$
  The candidate is eligible if and only if $\max_{k=1}^d (W'_{jk} - V'_{jk}) \le \theta$.
* **Dynamic Decision:**
  * If $\ge 1$ eligible boxes exist: Expand the box that minimizes total volume expansion $\sum_k (W'_{jk} - V'_{jk}) - (W_{jk} - V_{jk})$.
  * If 0 eligible boxes exist: Create a new micro-box $B_{\text{new}}$ with $\mathbf{V}_{\text{new}} = \mathbf{W}_{\text{new}} = \mathbf{x}_i$.
* **Time Complexity:** $\mathcal{O}(N \cdot H \cdot d)$, where $H$ is the number of boxes ($H \ll N$).

---

### Stage 3: Inference, Class Scoring & Centroid Tie-Breaking
* **Query Ingestion:** Query point $\mathbf{x} \in [0, 1]^d$.
* **Box Membership Evaluation:** Evaluated coordinate-wise across all $H$ boxes:
  $$\mu_j(\mathbf{x}) = 1.0 - \frac{1}{d} \sum_{k=1}^d \min(1.0, \max(0.0, \gamma (V_{jk} - x_k), \gamma (x_k - W_{jk})))$$
* **Class Max-Pooling:**
  $$M_c(\mathbf{x}) = \max_{j: \text{class}(B_j) = c} \mu_j(\mathbf{x}) \in [0, 1]$$
* **Centroid Tie-Breaker:**
  $$\mathcal{T}(\mathbf{x}) = \arg\max_{c \in \{0, \dots, C-1\}} M_c(\mathbf{x})$$
  $$\hat{y}(\mathbf{x}) = \begin{cases}
  c^* & \text{if } |\mathcal{T}(\mathbf{x})| = 1 \\
  \arg\min_{c \in \mathcal{T}(\mathbf{x})} \|\mathbf{x} - \mathbf{c}_c\|_2^2 & \text{if } |\mathcal{T}(\mathbf{x})| \ge 2 \text{ (Cross-Class Tie)}
  \end{cases}$$

---

### Stage 4: Multi-Angle Forensic Explanations
1. **PACE-SHAP (`explainers/shap_exact.py`):**
   * Computes exact Shapley attributions $\boldsymbol{\phi}(\mathbf{x}) \in \mathbb{R}^d$ using the Gray-code recurrence lattice ($K \cdot 2^d$ additions) for $d \le 12$.
   * Switches to 32 deterministic antithetic permutation pairs for $d > 12$.
   * Zero sampling variance ($r = 1.0000$).
2. **PACE-flip (`explainers/flip.py`):**
   * Uses floor/ceiling bounds pruning to eliminate $71\% - 98\%$ of boxes in $\mathcal{O}(1)$ time.
   * Sweeps 32 mid-quantiles to find the minimal single-feature perturbation flipping the decision in $0.05 - 1.4$ ms.
3. **PACE-Sparse Recourse (`explainers/counterfactual.py`):**
   * Locks immutable user features ($\delta_k = 0$ for $k \in \mathcal{L}$).
   * Uses combinatorial forward search to find the closest high-evidence target box, altering only $1 - 2$ actionable features.

---

### Stage 5: Intrinsic Trust Scoring & Misclassification Lie Detector
1. **Prediction Disagreement ($\text{PD}$):** Perturbs $\mathbf{x}$ across 50 spherical micro-nudges ($\sigma = 0.02$) and measures label flip rate.
2. **Explanation Instability ($\text{EI}$):** Measures mean cosine distance between original and perturbed PACE-SHAP attribution vectors.
3. **Box Evidence ($\text{BoxEv}$):** Measures normalized distance to authentic absorbed training samples.
4. **Corrected Uncertainty ($U^*$):** Assigns untied queries to $[0.0, 0.5]$ and overlapping ties to $[0.5, 1.0]$.
5. **PRS Aggregator:** Non-parametric quantile aggregation combining $\text{PD}, \text{EI}, \text{BoxEv},$ and $U^*$ into a single $[0.0, 1.0]$ trust score.
6. **Logistic Meta-Classifier:** Regularized model combining geometric signals with 10-NN disagreement and Jiang Trust Score to predict misclassification probability.

---

### Stage 6: Surgical Self-Repair (SR) & Active Learning (AL)
1. **Misclassification Detection:** Flags errors on holdout/validation rows.
2. **Proposition 1 Enforcement:** Confirms error is caused by overlapping cross-class boxes with membership $1.0$.
3. **Surgical Shrink:** Retracts the offending opposing-class box along the dimension of minimal violation.
4. **Target Micro-Expansion:** Bounded expansion of true-class box if within $\theta$.
5. **Active Learning (AL):** When a human annotation budget is available, selects unlabelled queries with minimal top-two margin ($M_{\text{top}} - M_{\text{second}}$).

---

### Stage 7: Validation Gate Safety Audit & Proposition 4 Spatial Certification
1. **Fast Cached Validation Gate:** Evaluates whether proposed surgical repair causes $\Delta \text{Macro-F1} < -0.5\%$ or $\Delta \text{Recall}_c < -1.0\%$.
   * If violated: Immediate $\mathcal{O}(1)$ pointer rollback.
   * If accepted: Edit committed.
2. **Proposition 4 Certificate Generation:**
   $$\mathcal{C}_{\text{spatial}} = \prod_{k=1}^d \left[ \min(V_{jk}, V'_{jk}) - \frac{1}{\gamma}, \, \max(W_{jk}, W'_{jk}) + \frac{1}{\gamma} \right]$$
3. **Audit Verification:** Produces mathematical proof that outside $\mathcal{C}_{\text{spatial}}$, 0 predictions across the universe are altered.

---

## 3. Complete Tensor Flow Architecture Across the Pipeline

| Stage | Input Tensor / Variable | Transformation Operator | Output Tensor / Variable | Mathematical Shape | Memory Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ingestion** | `X_raw` | Min-Max Scaler (Eq. 1) | `X` | `(N, d)` | `float64` |
| **Centroids** | `X`, `y` | Empirical Mean Grouping | `centroids` | `(C, d)` | `float64` |
| **Training** | `x_i`, `y_i` | Chebyshev Expansion (Eq. 2)| `boxes` ($V, W, c$) | `V: (H, d)`, `W: (H, d)` | `float64` |
| **Inference** | `X_query` | Box Penalty Kernel (Eq. 3)| `M_box` | `(n, H)` | `float64` |
| **Pooling** | `M_box` | Max-Reduce by Class | `M_class` | `(n, C)` | `float64` |
| **Ties** | `M_class` | Centroid Argmin (Eq. 4) | `y_pred` | `(n,)` | `int64` |
| **PACE-SHAP** | `X_query` | Recurrence Lattice | `phi` | `(n, d)` | `float64` |
| **PACE-flip** | `X_query` | Bounds Pruning + Quantiles | `flip_deltas` | `(n, d)` | `float64` |
| **Trust** | `X_query` | Perturbations + Distance | `PRS` | `(n,)` | `float64` |
| **Gate Audit**| `boxes_new` | Cached Margin Evaluation | `gate_status` | `bool` | In-Place |
| **Certificate**| `box_old`, `box_new`| Spatial Envelope (Prop 4) | `C_spatial` | `(d, 2)` | `float64` |
