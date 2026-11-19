# 08. Codebase Implementation Deep Dive: Core Modules

## 1. Overview of Core Implementation Architecture

This document provides a line-by-line, function-by-function, and algorithm-by-algorithm inspection of the primary algorithmic modules in `pace_cfmnn/`. Every implementation detail, loop invariant, tensor dimension, and optimization technique is documented directly from the repository source code.

```
+----------------------------------------------------------------------------------------------------+
|                                    CORE MODULE SOURCE CODE INVENTORY                               |
+--------------------------+---------------------------------------------------------+---------------+
| Source Code File         | Primary Responsibilities                                | Line Count    |
+--------------------------+---------------------------------------------------------+---------------+
| `cfmnn_core.py`          | CFMNN base model, ramp functions, training, inference   | 443 lines     |
| `explain.py`             | PACE-SHAP, PACE-flip, contrastive, decomposition        | 960 lines     |
| `counterfactual.py`      | Box-directed counterfactuals and PACE-sparse search     | 331 lines     |
| `flip_exact.py`          | Exact single-feature continuous flip sets               | 227 lines     |
| `gfmm_explain.py`        | Min-operator adaptation for GFMM hyperbox models        | 352 lines     |
| `decision.py`            | Exact Shapley values of the categorical decision rule   | 187 lines     |
| `hardness.py`            | #P-hardness reduction gadget from weighted voting games | 149 lines     |
+--------------------------+---------------------------------------------------------+---------------+
```

---

## 2. Deep Dive: `pace_cfmnn/cfmnn_core.py`

### 2.1 Low-Level Mathematical Primitives
Lines 38–49 implement the vectorized ramp and univariate feature membership terms:

```python
def ramp(t, gamma):
    """Simpson's ramp f(t) = max(0, 1 - max(0, gamma * min(1, t)))."""
    return np.maximum(0.0, 1.0 - np.maximum(0.0, gamma * np.minimum(1.0, t)))

def feature_terms(x, v, w, gamma):
    """Per-feature membership terms m_i = [f(x_i - w_i) + f(v_i - x_i)] / 2."""
    return 0.5 * (ramp(x - w, gamma) + ramp(v - x, gamma))
```

#### Engineering & Mathematical Optimizations
* **Zero Branching:** Evaluates purely via NumPy SIMD vector primitives (`np.maximum`, `np.minimum`), avoiding scalar Python conditionals.
* **Broadcasting Invariance:** Operates seamlessly across arbitrary tensor shapes: scalar, vector $(d,)$, matrix $(n, d)$, or 3D tensor $(n, H, d)$.
* **Floor Guarantee:** Because $v_{ji} \le w_{ji}$, input $x_i$ cannot simultaneously violate both lower and upper boundaries. Thus, one of the two terms is identically $1.0$, guaranteeing $m_{ji}(x_i) \in [0.5, 1.0]$.

---

### 2.2 Dynamic Class Buffer (`_ClassBuffer`)
Lines 76–109 implement an efficient memory manager for single-pass box creation:

```python
class _ClassBuffer:
    def __init__(self, d: int, capacity: int = 16):
        self.n = 0
        self.V = np.empty((capacity, d))
        self.W = np.empty((capacity, d))
        self.S = np.empty((capacity, d))
        self.N = np.empty(capacity, dtype=np.int64)
        self.order = np.empty(capacity, dtype=np.int64)
        self.members: list[list[int]] = []
```
* **Amortized Geometric Resizing:** When `self.n == len(self.N)`, arrays double in capacity ($16 \to 32 \to 64 \dots$), avoiding expensive repeated memory reallocations during training on large datasets (e.g., Covertype with 581,012 rows).

---

### 2.3 CFMNN Training Engine (`CFMNN.fit`)
Lines 147–248 implement Algorithm 2 from Aadarsh & Sai Prasad (2023) with our two architectural corrections:

```python
def fit(self, X, y):
    # Validation and scaling assertions
    X = np.asarray(X, dtype=float)
    if (X < -1e-9).any() or (X > 1.0 + 1e-9).any():
        raise ValueError("X must be scaled to [0, 1]")
    ...
```

#### Complete Algorithmic Logic (Pseudocode Mapped to Lines 170–215)
```
FOR each training row r = 0, ..., N-1:
    x = X[r], c = class_idx[r]
    IF class c has no boxes yet:
        Instantiate point box: V = x, W = x, S = x, N = 1, members = [r]
    ELSE:
        Compute membership b_j(x) for all existing boxes of class c
        
        IF x is inside best box (max b_j(x) == 1.0):
            IF centroid_rule == "fixed":
                S_best += x, N_best += 1, members_best.append(r)
            # IF centroid_rule == "original": point is skipped (flawed prior work)
            
        ELSE:
            IF expansion == "search":
                # Candidate search: rank boxes of class c by descending membership
                Find highest-membership box j that satisfies:
                    sum_i (max(W_ji, x_i) - min(V_ji, x_i)) <= d * theta
            ELSE (expansion == "best"):
                Test only the single highest-membership box
                
            IF valid expanding box j found:
                V_j = min(V_j, x)
                W_j = max(W_j, x)
                S_j += x, N_j += 1, members_j.append(r)
            ELSE:
                Instantiate new point box: V = x, W = x, S = x, N = 1, members = [r]
```

---

### 2.4 Prediction & Cache-Blocked Batch Inference (`CFMNN.predict_details`)
Lines 298–385 implement Algorithm 4 with cache-blocked chunking:

```python
def predict_details(self, X) -> Prediction:
    n, d = X.shape
    H = self.n_boxes_
    step = self._chunk_size()  # max(1, chunk_elems // (H * d))
    ...
```
* **Cache-Blocked Chunking:** The chunk size is dynamically calculated as:
  $$\text{step} = \max\left(1, \; \left\lfloor \frac{65,536}{H \cdot d} \right\rfloor \right)$$
  This guarantees that intermediate 3D working tensors $(n_{\text{chunk}}, H, d)$ occupy approximately $0.5$ MB of RAM, fitting comfortably inside CPU L2/L3 cache and avoiding memory bus latency.
* **Vectorized Tie-Breaking:**
  ```python
  # Top group extraction
  t = M >= M.max(axis=1, keepdims=True) - tol
  # Masked minimum distance selection
  winner = np.argmin(np.where(t, D, np.inf), axis=1)
  ```

---

## 3. Deep Dive: `pace_cfmnn/explain.py`

### 3.1 The `PaceShap` Architecture
Lines 388–760 implement the closed-form Shapley evaluation engine.

#### Exact Dynamic Recurrence Engine (`_explain_exact_recurrence`)
Added during the tough review audit to optimize exact subset evaluation from $O(K \cdot d \cdot 2^d)$ to $O(K \cdot 2^d)$:

```python
def subset_sums(base: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Subset sums for each row: base + D @ mask, computed in K * 2^d additions.
    D has shape (K, d), base shape (K,). Returns (K, 2^d)."""
    K, d = D.shape
    out = np.empty((K, 1 << d))
    out[:, 0] = base
    for i in range(d):
        span = 1 << i
        out[:, span:span + span] = out[:, :span] + D[:, i:i + 1]
    return out
```

#### Mathematical Verification of Recurrence
* For $i = 0$: `span = 1`. `out[:, 1:2] = out[:, 0:1] + D[:, 0:1]`, which is $b_j(z) + \Delta_{j0}$.
* For $i = 1$: `span = 2`. Generates subsets $\{1\}$ and $\{0, 1\}$ by adding $\Delta_{j1}$ to subsets $\emptyset$ and $\{0\}$.
* By induction, every subset sum is produced with exactly **one vector addition**, achieving a $2.13\times$ speedup over BLAS matrix multiplications.

#### Exact Bounding Box Pruning (Lines 658–700)
```python
# Compute lower and upper bounds across all possible 2^d subsets
lo = base + np.minimum(0.0, D).sum(axis=1)
hi = base + np.maximum(0.0, D).sum(axis=1)

# Pruning Rule 1: Box ceiling strictly below best floor in group
keep = hi >= (group_best_floor - 1e-12)

# Pruning Rule 2: Pairwise dominance check
diff_min = base_k - base_j + np.minimum(0.0, D_k - D_j).sum(axis=1)
if diff_min > 0:
    # Box k strictly dominates box j under all coalitions; drop box j
```

---

### 3.2 The `PaceFlip` Architecture
Lines 143–374 implement analytical single-feature occlusion.

#### Quantile Replacement Mechanics
Rather than iterating over all training instances, `PaceFlip` computes 32 mid-quantiles:
$$Q_{ik} = \text{Quantile}\left(Z_{\cdot, i}, \; \frac{k + 0.5}{32}\right), \quad k \in \{0, \dots, 31\}$$

#### Segmented Maximum Reduction (Lines 340–355)
Instead of executing nested Python loops over points, features, and quantiles, `PaceFlip` flattens surviving boxes into continuous segment buffers and calls NumPy's compiled `reduceat`:
```python
# Evaluate all replacement memberships in one vectorized block
G[has] = np.maximum.reduceat(Mt, start[has], axis=0)
sA, sO = G[0::2], G[1::2]  # Split class A and other-class maxima
```
This design achieves an execution latency of **$0.05$ to $1.4$ ms per explanation**, running $25\times$ to $134\times$ faster than batched brute-force occlusion.

---

## 4. Deep Dive: `pace_cfmnn/counterfactual.py`

### 4.1 Analytical Box Projection (`counterfactual`)
Lines 49–92 implement PACE box-directed counterfactual generation:

```python
def counterfactual(model: CFMNN, x, target, locked=None, ...):
    ...
    # Project x onto candidate target-class hyperboxes
    clipped = np.clip(x, V, W)
    clipped[:, locked] = x[locked]  # Respect immutable feature constraints
    dist = np.abs(clipped - x)[:, mut].sum(axis=1)
    ...
```

#### Step-by-Step Execution Sequence
1. **Filtering:** Discards target-class boxes whose coordinate intervals do not contain locked feature values ($V_{ki} \le x_i \le W_{ki}$ for all $i \in \text{locked}$).
2. **L1 Projection:** Projects $x$ onto the boundary faces of candidate boxes: $\tilde{x} = \text{clip}(x, V_k, W_k)$.
3. **Centroid Stepping:** If $\tilde{x}$ lands on an ambiguous tie boundary, steps along the interior vector towards centroid $\mu_k$:
   $$c(s) = \tilde{x} + s \cdot (\mu_k - \tilde{x}), \quad s \in \{0.25, 0.50, 0.75, 1.00\}$$
4. **Greedy Sparsification (`_sparsify`):** Iterates over changed features in ascending order of $|c_i - x_i|$, resetting feature $i$ back to $x_i$ and verifying that the model prediction remains the target class.

---

### 4.2 Combinatorial Sparse Search (`sparse_counterfactual`)
Lines 140–235 implement PACE-sparse (RQ1 extension B1):
* Evaluates subsets of feature changes sized $k = 1, 2, \dots, d$.
* Evaluates exact membership scores directly from CFMNN delta terms.
* Avoids model forward passes for unviable candidate sets.
* Outperforms DiCE in sparsity on 10 of 11 benchmark datasets (`results/extensions_v1/counterfactuals/`).

---

## 5. Deep Dive: Complexity Gadget (`pace_cfmnn/hardness.py`)

Lines 73–149 implement the formal polynomial reduction proving Proposition 5:

```python
def voting_game_model(weights, quota, gamma: float = 4.0) -> VotingGameModel:
    ...
    eps = 1.0 / (4.0 * n * int(w.max()))
    t = 2.0 * d * eps * w / gamma
    ...
```
* Generates 4 hyperboxes: $a_1, a_2$ (class 0) and $b_1, b_2$ (class 1).
* Sets coordinates such that weight features add exactly $\epsilon w_i$ to memberships $b_{a1}$ and $b_{b1}$.
* Margin difference evaluates to $2\epsilon \cdot \mathbb{I}(\sum_{i \in S} w_i \ge q)$.
* Directly verified by unit tests in `tests/test_hardness.py`.

---

## 6. Deep Dive: `pace_cfmnn/flip_exact.py`

Lines 40–180 implement Proposition 8 to calculate exact decision flip intervals:
* Finds critical boundary breakpoints: $\{v_{ji} - \frac{1}{\gamma}, v_{ji}, w_{ji}, w_{ji} + \frac{1}{\gamma}\} \cap [0, 1]$.
* Sorts breakpoints to establish open intervals where all membership functions are strictly linear affine.
* Solves linear intersection equations $m_A(t) = m_B(t)$ analytically to find exact flip thresholds without grid search.

---

## 7. Cross-References

- For the companion deep dive into training pipelines, active learning, and audit modules: [`09_Codebase_Implementation_Deep_Dive_Training_and_Utils.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/09_Codebase_Implementation_Deep_Dive_Training_and_Utils.md)
- For the end-to-end training pipeline and optimization strategy: [`10_Training_Pipeline_and_Optimization_Strategy.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/10_Training_Pipeline_and_Optimization_Strategy.md)
- For execution latency profiles across all core algorithms: [`16_Component_Wise_Testing_Time_Latency_and_Compute_Profile.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/16_Component_Wise_Testing_Time_Latency_and_Compute_Profile.md)
- For formal mathematical proofs of hardness and flip sets: [`21_Tough_Review_Audit_and_Formal_Proof_Extensions.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/21_Tough_Review_Audit_and_Formal_Proof_Extensions.md)
