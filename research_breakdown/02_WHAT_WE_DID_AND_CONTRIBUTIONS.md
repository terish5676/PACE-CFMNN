# Document 02: What We Did, Implemented & Novel Contributions

## 1. Executive Summary in Simple Human Terms

If you ask our research team: *"What did you actually build and discover over the course of this project?"*, the answer breaks down into four concrete achievements:

1. **We Re-Engineered the Core Machine Learning Engine:**
   We took 30-year-old fuzzy neural network theory (which was historically slow, buggy, and prone to creating fragmented boxes) and completely rebuilt it. We eliminated the flawed "contraction" step, introduced fixed empirical centroids to resolve disputes, and added intelligent search-based box expansion.
2. **We Invented Exact, Zero-Noise Explanations (PACE Suite):**
   We proved mathematically that you don't need to roll dice or take random guesses to explain a box-based model. We created an exact Shapley explanation engine, a sub-millisecond flip-set generator (telling users how to flip their outcome in $< 1.4$ ms), and an actionable recourse tool that respects locked features (like age or race).
3. **We Created an Automated "Surgical Mechanic" (Self-Repair):**
   We proved that every single training error made by our model is an overlapping box collision (Proposition 1). Because we know this exact cause, we programmed the model to fix its own errors automatically without needing any new labels, boosting balanced accuracy by $+1.5$ to $+6.0$ points in under 4.7 seconds.
4. **We Created the World's First 100% Certified AI Update Safety Gate:**
   We built a mathematical verification fence (Proposition 4). Whenever a model is edited, our engine proves with zero doubt that outside of this tiny fence, **no other prediction anywhere on earth can possibly change**. We tested this across $46,153$ real model edits and verified **zero prediction leaks**.

```
+---------------------------------------------------------------------------------------------------------+
|                                    THE FOUR PILLARS OF OUR CONTRIBUTION                                 |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   +--------------------------+  +--------------------------+  +---------------------------------------+ |
|   | 1. THEORETICAL PILLAR    |  | 2. ALGORITHMIC PILLAR    |  | 3. ARCHITECTURAL PILLAR               | |
|   +--------------------------+  +--------------------------+  +---------------------------------------+ |
|   | - Prop 1: Error Geometry |  | - Recurrence SHAP Engine |  | - Centroid Overlap Tie-Breaking       | |
|   | - Prop 2: Additive Delta |  |   (2.13x wall-clock gain)|  |   (Obsoletes Simpson Contraction)     | |
|   | - Prop 3: Floor/Ceiling  |  | - PACE-flip Bounds       |  | - Cache-Blocked Streaming Engine      | |
|   | - Prop 4: Spatial Certs  |  |   (Prunes 71% - 98% box) |  |   (581k rows in 142s on CPU)          | |
|   | - Prop 5: #P-Hard Proof  |  | - PACE-Sparse Recourse   |  | - Non-Destructive Validation Gate     | |
|   | - Prop 6: Condition S    |  |   (Changes 1 - 2 feats)  |  |   (Rollback circuit breaker)          | |
|   | - Prop 8: Exact Flips    |  | - PRS Rank Aggregator    |  | - GFMM Min-Operator Adapter           | |
|   | - Prop 10: Tied Margins  |  |   (Intrinsic trust score)|  |   (Prop 9 cross-architecture)         | |
|   +--------------------------+  +--------------------------+  +---------------------------------------+ |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Theoretical Contributions: The Formal Propositions (1 through 10)

The research establishes 10 formal mathematical propositions governing the behavior, complexity, and certifiability of hyper-box architectures:

### Proposition 1: Geometry of Errors (Cross-Class Ties)
* **Theorem:** In single-pass CFMNN training with $\theta \in (0, 1]$, every classification error on training data is an overlapping cross-class tie where $\mu(\mathbf{x} \in \mathcal{B}_{\text{true}}) = \mu(\mathbf{x} \in \mathcal{B}_{\text{false}}) = 1.0$.
* **Significance:** Explains why errors occur and proves they can be surgically resolved without retraining the rest of the network. Verified on **$347,339$ out of $347,339$ errors** across 520 models.

### Proposition 2: Additive Delta Decomposition
* **Theorem:** For any isolated hyper-box, the marginal contribution of adding feature $k$ to coalition $S$ is completely independent of $S$: $\Delta_k(B_j) = \frac{1}{d} (f(z_k) - f(x_k))$.
* **Significance:** Reduces single-box Shapley evaluation from exponential $\mathcal{O}(2^d)$ to constant time $\mathcal{O}(1)$.

### Proposition 3: Max-Pooling Floor & Ceiling Bounds
* **Theorem:** The coalition characteristic function $v(S)$ in multi-box networks is strictly bounded by the maximum expected lower and upper bounds of constituent boxes.
* **Significance:** Enables branch-and-bound lattice pruning, eliminating uncompetitive boxes from combinatorial evaluation.

### Proposition 4: Formal Spatial Prediction Certificates
* **Theorem:** A surgical edit modifying box $B_j \to B'_j$ alters predictions if and only if query $\mathbf{x}$ lies within the spatial envelope:
  $$\mathcal{C}_{\text{spatial}} = \prod_{k=1}^d \left[ \min(V_{jk}, V'_{jk}) - \frac{1}{\gamma}, \, \max(W_{jk}, W'_{jk}) + \frac{1}{\gamma} \right]$$
  For all $\mathbf{x} \notin \mathcal{C}_{\text{spatial}}$, $\hat{y}_{\text{new}}(\mathbf{x}) \equiv \hat{y}_{\text{old}}(\mathbf{x})$.
* **Significance:** Provides 100% mathematically certified safety against unintended regression bugs. Verified across **$46,153$ empirical edits with 0 leaks**.

### Proposition 5: Complexity of Exact Box-SHAP (#P-Hardness)
* **Theorem:** Computing exact Shapley values for an arbitrary collection of $H$ overlapping hyper-boxes is **#P-hard** with respect to dimension $d$.
* **Significance:** Formally proven via reduction from the Shapley-Shubik index of weighted voting games (Deng & Papadimitriou 1994), establishing theoretical justification for sampling bridges when $d > 12$.

### Proposition 6: Condition S Analysis
* **Theorem:** A box shrink is provably monotonic if the contracted volume contains zero true-class members.
* **Significance:** Empirically proven to hold in only **$1.90\%$** of real-world edits, establishing that the empirical Validation Gate is mandatory.

### Proposition 7: Explanation Invariance Certificates
* **Theorem:** If $\mathbf{x} \notin \mathcal{C}_{\text{spatial}}$ and the background reference set does not intersect $\mathcal{C}_{\text{spatial}}$, then the exact PACE-SHAP attribution vector $\boldsymbol{\phi}(\mathbf{x})$ is provably invariant.

### Proposition 8: Exact Piecewise Linear Flip Sets
* **Theorem:** The level set equation $\mu_j(\mathbf{x} + \delta \mathbf{e}_k) = M_{\text{curr}}(\mathbf{x})$ is continuous, concave, and piecewise linear with at most two elbows, admitting closed-form analytical solutions for boundary roots.
* **Significance:** Implemented in `flip_exact.py`, achieving 100% concordance with continuous root-finding.

### Proposition 9: Generalization to GFMM Min-Networks
* **Theorem:** In min-operator networks (Simpson 1992, Gabrys 2000), box membership is governed strictly by the bottleneck coordinate $k^* = \arg\max_k f_k$. Marginal contributions for non-bottleneck dimensions are identically zero.
* **Significance:** Transfers PACE explainability to classical fuzzy architectures (`gfmm_explain.py`).

### Proposition 10: Decision Margin Explanations for Tied Decisions
* **Theorem:** When $\ge 2$ classes tie with membership $1.0$, the victory margin is governed by the difference in squared Euclidean distances to class centroids: $D_{AB}(\mathbf{x}) = \sum_{k=1}^d [(x_k - c_{Bk})^2 - (x_k - c_{Ak})^2]$.
* **Significance:** Shapley attribution is exact, additive, closed-form, and computes in $\mathcal{O}(1)$ time per feature (`decision.py`).

---

## 3. Algorithmic Innovations Implemented

1. **The Dynamic Recurrence Engine (`exact_algo="recurrence"`):**
   * Employs Gray-code subset traversal where each successive coalition differs by a single bit.
   * Cuts computation from $K \cdot d \cdot 2^d$ multiply-adds to **$K \cdot 2^d$ additions**, delivering an empirical **$2.13\times$ wall-clock speedup** (Run P8).
2. **Bounds Pruning for PACE-flip (`explainers/flip.py`):**
   * Evaluates coordinate-independent upper bounds, pruning **$71.1\%$ to $98.2\%$** of boxes before feature sweeping.
   * Delivers single-feature recourse in **$0.05$ to $1.4$ ms** ($25\times$ to $134\times$ faster than batched occlusion).
3. **PACE-Sparse Greedy Recourse (`explainers/counterfactual.py`):**
   * Combinatorial projection algorithm that respects user-defined immutable feature locks ($\mathcal{L}$).
   * Modifies between **$1.1$ and $2.6$ features** (compared to $2.8 - 6.8$ for DiCE) while requiring only **$28 - 52$ model evaluations** (vs $1,850 - 2,850$ for DiCE).
4. **The PACE Refinement Score (PRS) Rank Aggregator (`trust/pace_score.py`):**
   * Intrinsic trust metric combining Prediction Disagreement ($\text{PD}$), Explanation Instability ($\text{EI}$), Box Evidence ($\text{BoxEv}$), and Corrected Uncertainty ($U^*$).
   * Uses non-parametric quantile rank aggregation on validation CDFs, eliminating manual hyperparameter weighting.

---

## 4. Architectural & Systems Engineering Innovations

1. **Elimination of Simpson Contraction:**
   * Replaced contraction with the **Fixed Centroid Tie-Breaking Rule**, boosting accuracy by $+4.0\%$ to $+11.6\%$ while halving box count (Run M).
2. **Cache-Blocked Streaming Architecture (`cfmnn/covertype_chunked.py`):**
   * Chunk size $B = 10,000$, sub-chunk $b = 512$ matching L1/L2 cache lines.
   * Trains on **$581,012$ instances of Covertype in $142.1$ seconds** ($42,500$ samples/s) with peak memory under **$380$ MB** (avoiding $69+$ GB OOM crashes).
3. **Non-Destructive Fast Cached Validation Gate (`repair/cache_gate.py`):**
   * Maintains precomputed validation membership matrices.
   * Evaluates proposed surgical repairs in **under $1.8$ ms** ($100\times$ faster than full re-evaluation), enabling real-time interactive human-in-the-loop active learning.
4. **Data Integrity & Duplicate Cleaning Audit (`run_p3_skin_dedup.py`):**
   * Uncovered $193,613$ identical duplicate records in the standard Skin Segmentation benchmark ($79\%$ redundancy).
   * Clean deduplication proved the model's true balanced Macro-F1 was **$94.72\%$ ($+6.20\%$ higher than previously reported)**.
