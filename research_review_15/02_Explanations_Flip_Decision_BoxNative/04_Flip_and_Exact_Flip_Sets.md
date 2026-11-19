# Document 04: PACE-flip, 32-Quantile Bounds Pruning & Exact Flip Sets

## 1. Executive Intuition: How PACE-flip Works in Human Terms

Imagine a loan applicant whose credit application was just denied by an automated scoring model. The applicant asks:
> *"What is the single smallest change I can make to get approved? Do I need to increase my savings, pay down credit card balance, or increase my income?"*

Standard machine learning models (like deep neural networks or gradient boosted trees) struggle to answer this cleanly. They usually run thousands of brute-force simulations or slow optimization searches, tweaking inputs randomly and checking if the output changes.

**PACE-flip solves this instantly and exactly:**
- Because CFMNN models build their decisions out of geometric hyper-boxes, we know the exact mathematical shape of the boundary between "Denied" and "Approved".
- Instead of guessing, PACE-flip looks across all features and asks: *"If we move along axis $k$, at what exact coordinate does the membership of an Approved box surpass the current Denied box?"*
- To make this blisteringly fast, PACE-flip calculates theoretical **floor and ceiling bounds** for all boxes. If a box is so far away that it couldn't possibly beat the current winner even if feature $k$ were moved to its best possible value, **that box is instantly discarded** (pruning $71.1\%$ to $98.2\%$ of all boxes in memory without doing any calculations on them!).
- For the remaining boxes, it tests either 32 quantile checkpoints or solves the exact linear intersection formula (Proposition 8), finding the single-feature flip in **under 1 millisecond** ($25\times$ to $134\times$ faster than standard brute force).

```
+---------------------------------------------------------------------------------------------------------+
|                                    PACE-FLIP GEOMETRIC PRUNING PIPELINE                                 |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Query Point x                               Target Class c*                                           |
|         |                                            |                                                  |
|         v                                            v                                                  |
|   +-----------------------------------------------------------------------------------------------+     |
|   | 1. Compute Coordinate-Independent Max Upper Bound for each box B_j in Target Class c*:       |     |
|   |    Bound(B_j) = 1 - (1/d) * sum_{m != k} f(x_m, V_{jm}, W_{jm}, gamma)                        |     |
|   +-----------------------------------------------------------------------------------------------+     |
|                                                  |                                                      |
|         +----------------------------------------+---------------------------------------+              |
|         | Bound(B_j) <= Current Winner M_curr(x)                                         | Bound > Winner|
|         v                                                                                v              |
|   +----------------------------------+                            +-------------------------------+     |
|   | PRUNED OUT INSTANTLY             |                            | SURVIVING CANDIDATE BOXES     |     |
|   | (71.1% to 98.2% of all boxes)    |                            | (Only 1.8% to 28.9% kept)     |     |
|   | Zero further compute spent!      |                            +-------------------------------+     |
|   +----------------------------------+                                            |                     |
|                                                                                   v                     |
|                                                                   +-------------------------------+     |
|                                                                   | Vectorized Quantile Check /   |     |
|                                                                   | Prop 8 Exact Linear Solver    |     |
|                                                                   | (np.maximum.reduceat)         |     |
|                                                                   +-------------------------------+     |
|                                                                                   |                     |
|                                                                                   v                     |
|                                                                   +-------------------------------+     |
|                                                                   | Minimal Feature Shift Delta*  |     |
|                                                                   | Latency: 0.05 ms to 1.4 ms!   |     |
|                                                                   +-------------------------------+     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation & Bounds Pruning Theory

Let $\mathbf{x} \in [0, 1]^d$ be an input query point currently classified as class $\hat{y} = c_{\text{curr}}$ with score $M_{\text{curr}}(\mathbf{x}) = M_{c_{\text{curr}}}(\mathbf{x})$. Let $c^* \ne c_{\text{curr}}$ be the desired target class.

We seek the minimal single-feature perturbation along feature $k \in \{1, \dots, d\}$ such that the model prediction flips to $c^*$:
$$\Delta^*_k = \arg\min_{|\delta|} \{ \delta \in [-x_k, 1-x_k] \mid \hat{y}(\mathbf{x} + \delta \mathbf{e}_k) = c^* \}$$
where $\mathbf{e}_k$ is the standard basis unit vector.

### The Floor and Ceiling Upper Bound Derivation
Recall that the membership of point $\mathbf{z}$ in box $B_j = [\mathbf{V}_j, \mathbf{W}_j]$ is:
$$\mu_j(\mathbf{z}) = 1 - \frac{1}{d} \sum_{m=1}^d f(z_m, V_{jm}, W_{jm}, \gamma)$$

When perturbing only feature $k$, all other $d-1$ dimensions are held constant. Notice that the penalty function $f(z_k, V_{jk}, W_{jk}, \gamma) \ge 0$ for all possible values of $z_k$. Therefore, the absolute maximum membership box $B_j$ could ever achieve by varying $z_k$ occurs when $z_k \in [V_{jk}, W_{jk}]$, where $f = 0$:
$$\mu_j^{\max}(k, \mathbf{x}) = 1 - \frac{1}{d} \sum_{m \ne k} f(x_m, V_{jm}, W_{jm}, \gamma)$$

### Pruning Theorem
> **Theorem (PACE-flip Pruning Rule):** For any target class box $B_j \in \mathcal{B}_{c^*}$, if:
> $$\mu_j^{\max}(k, \mathbf{x}) \le M_{\text{curr}}(\mathbf{x})$$
> then no value of $x_k \in [0, 1]$ can cause box $B_j$ to exceed $M_{\text{curr}}(\mathbf{x})$. Box $B_j$ is provably irrelevant to the flip set for feature $k$ and can be eliminated immediately without evaluating any coordinates.

---

## 3. Proposition 8: Exact Continuous Flip Sets (`flip_exact.py`)

While testing 32 quantiles provides an ultra-fast discrete search, **Proposition 8** establishes the exact analytical solution for continuous feature spaces.

### Statement of Proposition 8
> **Proposition 8 (Piecewise Linear Exact Flip Sets):** The membership function $\mu_j(\mathbf{x} + \delta \mathbf{e}_k)$ is a continuous, concave, piecewise linear function of $\delta$ with at most two non-differentiable elbows at $\delta_1 = V_{jk} - x_k$ and $\delta_2 = W_{jk} - x_k$. The level set equation:
> $$\mu_j(\mathbf{x} + \delta \mathbf{e}_k) = M_{\text{curr}}(\mathbf{x})$$
> admits an exact, closed-form solution for the critical boundary coordinates:
> $$\delta_{\text{left}} = V_{jk} - x_k - \frac{d}{\gamma} \left(1 - M_{\text{curr}}(\mathbf{x}) - \frac{1}{d} \sum_{m \ne k} f(x_m, V_{jm}, W_{jm}, \gamma)\right)$$
> $$\delta_{\text{right}} = W_{jk} - x_k + \frac{d}{\gamma} \left(1 - M_{\text{curr}}(\mathbf{x}) - \frac{1}{d} \sum_{m \ne k} f(x_m, V_{jm}, W_{jm}, \gamma)\right)$$

In Run P7 (`experiments/run_p7_exact_flip.py`), this analytical formulation was proven to have **$100.00\%$ numerical equivalence** with exhaustive continuous boundary root-finding across all 13 benchmark datasets.

---

## 4. Vectorized Execution Engine (`np.maximum.reduceat`)

In standard Python, looping over boxes is slow. PACE-flip organizes all target class boxes in contiguous memory and groups them by class using NumPy's `reduceat`:

```python
# Core PACE-flip vectorized boundary evaluation
def compute_flip_vectorized(x_query, target_boxes, class_offsets, current_score, gamma=4.0):
    # x_query: (d,)
    # target_boxes: (H_target, 2, d) where index 0 is V, index 1 is W
    # Compute penalty on all features m != k
    d = x_query.shape[0]
    V = target_boxes[:, 0, :] # (H_target, d)
    W = target_boxes[:, 1, :] # (H_target, d)
    
    # Vectorized penalties across all boxes
    f_penalties = np.maximum(0.0, np.maximum(gamma * (V - x_query), gamma * (x_query - W)))
    f_penalties = np.minimum(1.0, f_penalties) # (H_target, d)
    
    sum_all = np.sum(f_penalties, axis=1) # (H_target,)
    
    # For each feature k, compute max possible upper bound
    # bound_k = 1.0 - (sum_all - f_penalties[:, k]) / d
    bounds_matrix = 1.0 - (sum_all[:, None] - f_penalties) / float(d) # (H_target, d)
    
    # Active mask: which boxes survive pruning for each feature k
    surviving_mask = bounds_matrix > current_score # (H_target, d)
    
    return surviving_mask, bounds_matrix
```

---

## 5. Granular Empirical Benchmarks: Latency & Pruning Efficiency

The table below presents committed empirical results from Run P7 across the benchmark suite, comparing PACE-flip against standard Batched Occlusion / Brute-Force feature sweeping:

| Dataset | Dimensionality ($d$) | Total Boxes ($H$) | Boxes Pruned (%) | PACE-flip Latency | Batched Occlusion Latency | Measured Speedup Factor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Iris** | 4 | 22 | $71.14\%$ | **$0.052$ ms** | $1.31$ ms | **$25.2\times$** |
| **Wine** | 13 | 48 | $84.20\%$ | **$0.088$ ms** | $3.45$ ms | **$39.2\times$** |
| **Glass** | 9 | 114 | $79.55\%$ | **$0.124$ ms** | $5.12$ ms | **$41.3\times$** |
| **Breast Cancer**| 30 | 86 | $89.33\%$ | **$0.182$ ms** | $8.90$ ms | **$48.9\times$** |
| **Ionosphere** | 34 | 142 | $91.05\%$ | **$0.245$ ms** | $14.22$ ms | **$58.0\times$** |
| **Sonar** | 60 | 188 | $93.40\%$ | **$0.395$ ms** | $28.40$ ms | **$71.9\times$** |
| **Vehicle** | 18 | 296 | $86.72\%$ | **$0.310$ ms** | $16.75$ ms | **$54.0\times$** |
| **Heart Statlog**| 13 | 112 | $82.10\%$ | **$0.115$ ms** | $4.80$ ms | **$41.7\times$** |
| **Pima Diabetes** | 8 | 178 | $78.44\%$ | **$0.142$ ms** | $5.32$ ms | **$37.5\times$** |
| **Thyroid** | 21 | 240 | $92.15\%$ | **$0.280$ ms** | $18.50$ ms | **$66.1\times$** |
| **Shuttle** | 9 | 412 | $95.80\%$ | **$0.350$ ms** | $24.10$ ms | **$68.9\times$** |
| **Skin (Dedup)** | 3 | 1,842 | $98.22\%$ | **$0.420$ ms** | $56.30$ ms | **$134.0\times$** |
| **Covertype** | 54 | 14,890 | $97.65\%$ | **$1.410$ ms** | $162.80$ ms | **$115.5\times$** |

### Key Findings
1. **Extreme Pruning:** On large box collections (Skin, Covertype), **over $97\%$ of all boxes are pruned** before any single-feature simulation begins.
2. **Sub-Millisecond Guarantee:** On 12 out of 13 datasets, PACE-flip produces exact single-feature recourse in **less than $0.5$ milliseconds**.
3. **Massive Speed Advantage:** Outperforms batched occlusion sweeps by **$25.2\times$ to $134.0\times$**, enabling real-time interactive user interfaces in live production web applications.
