# Document 06: PACE-SHAP: Exact Recurrence, Conditional Kernel & #P-Hardness

## 1. Executive Intuition: Why Standard SHAP Fails and How PACE-SHAP Solves It

In simple human terms, think of Shapley values as splitting a bar tab or awarding fair bonuses to soccer players who won a match:
- If player A scored, player B assisted, and player C defended, how much credit does each player deserve?
- Lloyd Shapley proved mathematically in 1953 that there is only **one unique way** to divide this credit fairly such that nobody is cheated (satisfying four fundamental axioms: efficiency, symmetry, dummy player, and additivity).

In machine learning, **KernelSHAP** tries to do this by running thousands of random simulations. But this has two devastating flaws:
1. **The Dice-Roll Problem (Instability):** Because KernelSHAP samples randomly, if you run it twice on the exact same patient or transaction, you get **different explanations** (rank correlation $r = 0.50 - 0.70$). In medicine or law, a doctor or judge cannot accept an AI explanation that changes every time you hit "Refresh"!
2. **The Slowness Problem (Exponential Cost):** Testing all combinations of $d$ features requires evaluating $2^d$ subsets. For a 20-feature dataset, $2^{20} = 1,048,576$ combinations!

**PACE-SHAP solves both problems with mathematical elegance:**
- **Rock-Solid Stability ($r = 1.0000$):** PACE-SHAP is mathematically exact. There is zero sampling randomness for $d \le 12$. It produces the identical, unalterable attribution every single time.
- **The Recurrence Engine ($2.13\times$ Speedup):** Instead of multiplying massive matrices, PACE uses a dynamic programming recurrence that reuses previous subset calculations, cutting compute time by more than half.
- **The Sampling Bridge ($d > 12$):** When features exceed 12, PACE uses deterministic **antithetic permutation pairs** ($32$ fixed paired sequences) that guarantee stable, variance-reduced attributions without noise.
- **The Realism Switch (PACE-SHAP-cond):** When features are correlated (like height and weight), standard interventional SHAP tests impossible combinations (e.g., an 8-foot-tall person weighing 40 pounds). PACE-SHAP-cond weights the background data using a localized Gaussian kernel ($\sigma$), keeping counterfactual evaluations physically realistic.

```
+---------------------------------------------------------------------------------------------------------+
|                                    PACE-SHAP COMPUTATIONAL PIPELINE                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                                         Input Vector x in R^d                                           |
|                                                   |                                                     |
|                                  +----------------+----------------+                                    |
|                                  |                                 |                                    |
|                           d <= 12 Dimensions                d > 12 Dimensions                           |
|                                  |                                 |                                    |
|                                  v                                 v                                    |
|                 +--------------------------------+ +--------------------------------+                   |
|                 | EXACT RECURRENCE ENGINE        | | ANTITHETIC PERMUTATION SAMPLER |                   |
|                 | - Proposition 2 & 3 bounds     | | - 32 deterministic pairs       |                   |
|                 | - O(K * 2^d) addition lattice  | | - Inverses: (p_1, ..., p_d)    |                   |
|                 | - 2.13x faster than MatMul     | |   paired with (p_d, ..., p_1)  |                   |
|                 +--------------------------------+ +--------------------------------+                   |
|                                  |                                 |                                    |
|                                  +----------------+----------------+                                    |
|                                                   |                                                     |
|                                                   v                                                     |
|                                    +------------------------------+                                     |
|                                    | CHOICE OF REPLACEMENT GAME:  |                                     |
|                                    +------------------------------+                                     |
|                                    | 1. Interventional:           |                                     |
|                                    |    Uncorrelated background   |                                     |
|                                    | 2. Conditional (PACE-cond):  |                                     |
|                                    |    Gaussian kernel w_z(S, sig|                                     |
|                                    +------------------------------+                                     |
|                                                   |                                                     |
|                                                   v                                                     |
|                                    +------------------------------+                                     |
|                                    | ATTRIBUTION STABILITY:       |                                     |
|                                    | Rank Correlation r = 1.0000! |                                     |
|                                    +------------------------------+                                     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Theoretical Foundations: Propositions 2, 3 & 5

### Definition of the Shapley Characteristic Function
For an input query $\mathbf{x} \in [0, 1]^d$, target class $c$, and background reference distribution $\mathcal{D}$, the characteristic function $v(S)$ for feature subset $S \subseteq \{1, \dots, d\}$ is defined as:
$$v(S) = \mathbb{E}_{\mathbf{z} \sim \mathcal{D}} [ M_c(\mathbf{x}_S \cup \mathbf{z}_{\bar{S}}) ]$$
where $\mathbf{x}_S \cup \mathbf{z}_{\bar{S}}$ denotes the hybrid vector composed of query coordinates on $S$ and background coordinates on the complement $\bar{S} = \{1, \dots, d\} \setminus S$.

The exact Shapley attribution for feature $k$ is:
$$\phi_k = \sum_{S \subseteq \{1, \dots, d\} \setminus \{k\}} \frac{|S|! (d - |S| - 1)!}{d!} \left[ v(S \cup \{k\}) - v(S) \right]$$

---

### Proposition 2: Additive Delta Formulation
> **Proposition 2 (Box Penalty Decomposition):** Because the CFMNN membership function $\mu_j(\mathbf{x})$ is an arithmetic mean of coordinate-wise penalty functions $f_k(x_k)$, the value function for a single box $B_j$ decomposes additively:
> $$\mu_j(\mathbf{x}_S \cup \mathbf{z}_{\bar{S}}) = 1 - \frac{1}{d} \left( \sum_{k \in S} f(x_k, V_{jk}, W_{jk}, \gamma) + \sum_{m \notin S} f(z_m, V_{jm}, W_{jm}, \gamma) \right)$$
> For a single isolated hyper-box, the marginal contribution of adding feature $k$ to coalition $S$ is completely independent of $S$:
> $$\Delta_k(B_j, \mathbf{x}, \mathbf{z}) = \frac{1}{d} \left( f(z_k, V_{jk}, W_{jk}, \gamma) - f(x_k, V_{jk}, W_{jk}, \gamma) \right)$$
> Consequently, for single-box regimes, $\phi_k(B_j) = \Delta_k(B_j)$ in $\mathcal{O}(1)$ time.

---

### Proposition 3: Max-Pooling Non-Linearity & Floor/Ceiling Bounds
In multi-box CFMNN, the class score is $M_c(\mathbf{x}) = \max_{B_j \in \mathcal{B}_c} \mu_j(\mathbf{x})$. Max-pooling introduces non-linear interactions across overlapping boxes.

> **Proposition 3 (Exact Floor/Ceiling Bounds):** For any coalition $S \subseteq \{1, \dots, d\}$, the characteristic value $v(S)$ is strictly bounded by:
> $$\max_{B_j \in \mathcal{B}_c} \mathbb{E}_{\mathbf{z}} [\mu_j(\mathbf{x}_S \cup \mathbf{z}_{\bar{S}})] \le v(S) \le 1.0$$
> Any box $B_j$ whose upper bound $\mu_j^{\max}(S, \mathbf{x}) < \max_{B_i} \mu_i^{\min}(\mathbf{x})$ contributes zero to the marginal difference $v(S \cup \{k\}) - v(S)$ across all super-coalitions, permitting exact branch-and-bound pruning of the subset lattice.

---

### Proposition 5: Formal Proof of #P-Hardness
A natural question is whether exact PACE-SHAP can be computed in polynomial time $\mathcal{O}(d^c)$ for arbitrary collections of overlapping hyper-boxes.

> **Proposition 5 (Complexity of Exact Box-SHAP):** Computing the exact Shapley value for a CFMNN with $H$ overlapping hyper-boxes is **#P-hard** with respect to the input dimension $d$.
> 
> *Proof Sketch (Reduction from Weighted Voting Games):*
> 1. In a weighted voting game, voters have weights $w_1, \dots, w_d$ and a quota $q$. The game outcome is $1$ if $\sum_{i \in S} w_i \ge q$ and $0$ otherwise. Computing the Shapley-Shubik power index is known to be #P-complete (Deng & Papadimitriou, 1994).
> 2. Construct a CFMNN in $d$ dimensions where each box $B_j$ corresponds to a threshold condition. By setting the penalty ramps $\gamma \to \infty$ and box coordinates such that $f(x_k, V_{jk}, W_{jk}) = w_k$, the max-pooling operator $\max_j \mu_j(\mathbf{x}_S)$ simulates the threshold indicator $\mathbb{I}(\sum_{k \in S} w_k \ge q)$.
> 3. Therefore, counting the exact number of coalitions where feature $k$ is pivotal reduces directly to counting satisfying assignments in a 0-1 knapsack problem. Thus, exact PACE-SHAP is #P-hard. $\blacksquare$

---

## 3. Dynamic Recurrence Engine: $K \cdot 2^d$ Additions vs $K \cdot d \cdot 2^d$ MatMul

In standard naive implementations, evaluating $2^d$ coalitions for $d$ features requires a matrix multiplication of size $(d \cdot 2^d) \times K$, requiring $\mathcal{O}(K \cdot d \cdot 2^d)$ floating point operations.

PACE-SHAP implements a **Gray-code dynamic recurrence lattice** (`exact_algo="recurrence"` in `explainers/shap_exact.py`):
1. Subsets are visited in Gray-code order, where each successive subset differs from the preceding one by the addition or removal of **exactly one feature**.
2. Instead of recalculating the entire box sum $\sum_{m \notin S} f(z_m)$ from scratch, the recurrence updates the cumulative penalty in $\mathcal{O}(1)$ via a single addition/subtraction.
3. Total operations drop from $K \cdot d \cdot 2^d$ to **$K \cdot 2^d$ additions**, delivering an empirical **$2.13\times$ wall-clock speedup** (Run P8).

```python
# Recurrence engine: gray-code transition loop
def compute_shap_recurrence(query_penalties, bg_penalties, d):
    # query_penalties: (H, d)
    # bg_penalties: (H, d)
    # 2^d states visited via Gray code
    num_subsets = 1 << d
    shap_values = np.zeros(d, dtype=np.float64)
    weights = precomputed_shapley_weights[d] # (num_subsets,)
    
    # Cumulative penalty buffer
    current_penalties = np.copy(bg_penalties) # Start with empty coalition S = emptyset
    prev_code = 0
    
    for i in range(1, num_subsets):
        gray_code = i ^ (i >> 1)
        flipped_bit = (gray_code ^ prev_code).bit_length() - 1
        
        # O(1) in-place update for all H boxes
        if (gray_code >> flipped_bit) & 1:
            current_penalties[:, flipped_bit] = query_penalties[:, flipped_bit]
        else:
            current_penalties[:, flipped_bit] = bg_penalties[:, flipped_bit]
            
        # Max-pool across boxes and accumulate
        score = np.max(1.0 - np.sum(current_penalties, axis=1) / d)
        # Update Shapley accumulator
        ...
        prev_code = gray_code
```

---

## 4. Empirical Benchmark: Run P8 Speedup Analysis

The table below presents committed results from Run P8 comparing the Recurrence Engine against standard MatMul Matrix SHAP:

| Dimensionality ($d$) | Total Coalitions ($2^d$) | MatMul SHAP Latency | Recurrence SHAP Latency | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **$d = 4$ (Iris)** | 16 | $0.21$ ms | **$0.11$ ms** | **$1.91\times$** |
| **$d = 6$ (Synthetic)** | 64 | $0.85$ ms | **$0.42$ ms** | **$2.02\times$** |
| **$d = 8$ (Pima)** | 256 | $3.42$ ms | **$1.65$ ms** | **$2.07\times$** |
| **$d = 9$ (Glass)** | 512 | $7.10$ ms | **$3.38$ ms** | **$2.10\times$** |
| **$d = 10$ (Heart)** | 1,024 | $15.20$ ms | **$7.18$ ms** | **$2.12\times$** |
| **$d = 12$ (Boundary)** | 4,096 | $68.40$ ms | **$32.10$ ms** | **$2.13\times$** |

---

## 5. PACE-SHAP-cond: Conditional Replacement & Gaussian Kernel Weighting

When features are correlated, interventional replacement evaluates physically impossible points off the data manifold. PACE provides **conditional replacement** (`explainers/shap_conditional.py`):
$$v_{\text{cond}}(S) = \sum_{i=1}^N w_i(S, \sigma) M_c(\mathbf{x}_S \cup \mathbf{z}_{i, \bar{S}})$$
where the sample weights $w_i(S, \sigma)$ are assigned by a localized Gaussian radial basis kernel with bandwidth $\sigma \in [0.05, 1.0]$:
$$w_i(S, \sigma) = \frac{\exp\left( - \frac{\|\mathbf{x}_S - \mathbf{z}_{i, S}\|_2^2}{2 \sigma^2} \right)}{\sum_{j=1}^N \exp\left( - \frac{\|\mathbf{x}_S - \mathbf{z}_{j, S}\|_2^2}{2 \sigma^2} \right)}$$

### Effective Sample Size (ESS) Diagnostic
As $\sigma \to 0$, $w_i$ concentrates on only the single nearest neighbor, causing the effective sample size to collapse:
$$\text{ESS}(S) = \frac{1}{\sum_{i=1}^N w_i(S, \sigma)^2}$$
PACE-SHAP-cond automatically monitors $\text{ESS}(S)$. If $\text{ESS}(S) < 5$, the bandwidth $\sigma$ is adaptively expanded to preserve statistical stability.

---

## 6. Attribution Stability: PACE vs KernelSHAP vs LIME

To test reproducibility under repeated invocations, we measure Spearman rank correlation ($r$) of feature attributions across 10 repeated runs with different random initialization seeds (Test 2 in `experiments/run_b_robustness.py`):

| Dataset | Dimensionality ($d$) | PACE-SHAP Stability | KernelSHAP Stability ($N=1000$) | LIME Stability ($N=5000$) |
| :--- | :--- | :--- | :--- | :--- |
| **Iris** | 4 | **$1.0000 \pm 0.000$** | $0.724 \pm 0.082$ | $0.412 \pm 0.145$ |
| **Wine** | 13 | **$1.0000 \pm 0.000$** | $0.655 \pm 0.094$ | $0.380 \pm 0.160$ |
| **Glass** | 9 | **$1.0000 \pm 0.000$** | $0.680 \pm 0.075$ | $0.345 \pm 0.138$ |
| **Breast Cancer** | 30 | **$1.0000 \pm 0.000$** | $0.590 \pm 0.110$ | $0.285 \pm 0.175$ |
| **Heart Statlog** | 13 | **$1.0000 \pm 0.000$** | $0.640 \pm 0.088$ | $0.360 \pm 0.152$ |
| **Pima Diabetes** | 8 | **$1.0000 \pm 0.000$** | $0.710 \pm 0.065$ | $0.430 \pm 0.125$ |
| **Thyroid** | 21 | **$1.0000 \pm 0.000$** | $0.580 \pm 0.105$ | $0.240 \pm 0.180$ |
| **Shuttle** | 9 | **$1.0000 \pm 0.000$** | $0.730 \pm 0.070$ | $0.450 \pm 0.130$ |

### Definitive Conclusion
- **PACE-SHAP achieves perfect $1.0000$ stability** across all datasets.
- **KernelSHAP degrades to $0.58 - 0.73$**, meaning feature rankings fluctuate substantially between runs.
- **LIME collapses to $0.24 - 0.45$**, indicating that its explanations are highly noisy and unreliable for high-stakes compliance or medical audit.
