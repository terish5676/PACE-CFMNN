# 10. Training Pipeline and Optimization Strategy

## 1. The Non-Gradient Training Paradigm

Unlike standard artificial neural networks that optimize continuous weight tensors via stochastic gradient descent (SGD, Adam) and backpropagation over multiple epochs, the Centroid-Based Fuzzy Min-Max Neural Network (CFMNN) operates under an entirely different computational paradigm:
* **Single-Pass Incremental Learning:** The model processes each training instance $(x_r, y_r)$ exactly once ($1$ epoch).
* **Deterministic Geometric Aggregation:** Instead of continuous loss minimization, learning consists of discrete, bounded spatial expansion of hyperboxes and running centroid vector updates.
* **Zero Gradient Descent Overhead:** There is no learning rate, no momentum, no weight decay, no backward pass, and no loss function backpropagation.

```
+----------------------------------------------------------------------------------------------------+
|                                    CFMNN SINGLE-PASS TRAINING FLOW                                 |
|                                                                                                    |
|    Input Instance (x_r, y_r) presented sequentially (r = 1, ..., N)                               |
|                     |                                                                              |
|                     v                                                                              |
|    Does a box of class y_r exist?                                                                  |
|         |                                                                                          |
|         +---[ NO ]---> Instantiate Point Box B_new:                                                |
|         |              V = x_r, W = x_r, mu = x_r, N = 1, members = [r]                            |
|         |                                                                                          |
|         v [ YES ]                                                                                  |
|    Evaluate membership b_j(x_r) for all boxes j of class y_r                                       |
|    Identify best same-class box: k* = argmax b_j(x_r)                                              |
|         |                                                                                          |
|         v                                                                                          |
|    Is x_r already inside box k* (b_k*(x_r) == 1.0)?                                                |
|         |                                                                                          |
|         +---[ YES ]--> Centroid Rule Decision:                                                     |
|         |              - Fixed: S_k* += x_r, N_k* += 1, mu_k* = S/N, members.append(r)            |
|         |              - Original: Skip point entirely (flawed paper rule)                         |
|         |                                                                                          |
|         v [ NO ]                                                                                   |
|    Can an existing same-class box expand to absorb x_r?                                            |
|    Expansion Condition: sum_i (max(W_ji, x_ri) - min(V_ji, x_ri)) <= d * theta                    |
|         |                                                                                          |
|         +---[ YES ]--> Expand candidate box j*:                                                    |
|         |              V_j* = min(V_j*, x_r), W_j* = max(W_j*, x_r)                                |
|         |              S_j* += x_r, N_j* += 1, mu_j* = S/N, members.append(r)                     |
|         |                                                                                          |
|         +---[ NO ]---> Instantiate Point Box B_new:                                                |
|                        V = x_r, W = x_r, mu = x_r, N = 1, members = [r]                            |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Mechanics of Training Operations

### 2.1 Hyperbox Size Budget & Bounded Expansion
The hyperbox maximum size is constrained by a global hyperparameter $\theta \in (0, 1]$. For an existing hyperbox with lower corner $v_j$ and upper corner $w_j$, absorbing a new instance $x$ requires expanding its bounds to:
$$v'_{ji} = \min(v_{ji}, \, x_i), \qquad w'_{ji} = \max(w_{ji}, \, x_i)$$
The expansion is permitted if and only if the sum of edge lengths across all $d$ dimensions satisfies the size budget:
$$\sum_{i=1}^d \big(w'_{ji} - v'_{ji}\big) \le d \cdot \theta$$

#### Geometric Interpretation
* $\theta$ represents the maximum average allowable edge length of any hyperbox.
* When $\theta$ is small (e.g., $\theta = 0.05$), boxes remain tiny, leading to many localized clusters and higher risk of overfitting/memorization.
* When $\theta$ is large (e.g., $\theta = 0.60$), boxes expand to engulf large regions, resulting in high overlap across classes.

---

### 2.2 Expansion Policies: `"search"` vs. `"best"`

The codebase implements two distinct policies for selecting which same-class box expands:

1. **The Paper's Policy (`expansion="best"`):**
   - Evaluates only the single hyperbox achieving the highest membership score $b_j(x)$ for class $y$.
   - If that single box violates the expansion budget $\sum_i (w'_{ji} - v'_{ji}) > d \cdot \theta$, the algorithm immediately abandons expansion and creates a new point box.
   - *Failure Mode:* In moderate-to-high dimensions ($d \ge 10$), the best box is almost always at capacity. Consequently, almost all training instances spawn degenerate point boxes that never grow, causing the box count to explode ($3,748$ boxes for $4,000$ points on synthetic S1).

2. **The Simpson-Style Search Policy (`expansion="search"`, Default):**
   - Ranks all existing hyperboxes of class $y$ in descending order of membership $b_j(x)$.
   - Scans down the ranked list to find the first hyperbox that can absorb $x$ while remaining within budget $\theta$.
   - *Advantage:* Allows smaller, non-maximal boxes of the same class to absorb the point, preventing box proliferation and dramatically improving accuracy ($81.7\% \to 95.3\%$ on Digits, $95.0\% \to 97.2\%$ on Thyroid).

---

### 2.3 Centroid Accumulation: `"fixed"` vs. `"original"`

The internal centroid $\mu_j$ represents the empirical prototype of box $j$:
$$\mu_j = \frac{1}{N_j} \sum_{r \in \text{members}(j)} x_r$$

1. **The Original Paper's Rule (`centroid_rule="original"`):**
   - In Aadarsh & Sai Prasad (2023), Algorithm 2 lines 9–10 explicitly stated: if an input point already falls inside its best same-class box ($b_j(x) = 1.0$), it is skipped.
   - *The Flaw:* Points that fall inside an existing box do not update $N_j$ or coordinate sum $S_j$. As a result, the centroid represents only the initial point that created the box plus the few points on the boundary that expanded it.
   - *Consequence:* On Shuttle, where $99.5\%$ of points land inside existing boxes, centroids were severely unrepresentative. When boxes tied during inference, the centroids made catastrophic decisions, yielding $93.33\%$ accuracy.

2. **The Fixed Centroid Rule (`centroid_rule="fixed"`, Default):**
   - Every single point absorbed by a box updates its count $N_j \leftarrow N_j + 1$ and sum vector $S_j \leftarrow S_j + x$.
   - Centroids represent the true arithmetic mean of all absorbed points.
   - *Result:* Resolves tie-breaking anomalies, raising Shuttle accuracy from $93.33\%$ to $99.80\%$.

---

## 3. Training Order Sensitivity & Theoretical Complexity

### 3.1 Why Presentation Order Matters
Because CFMNN is a single-pass greedy clustering algorithm without backpropagation, the order in which training rows are presented dictates:
1. Which training instances instantiate new hyperbox seeds.
2. Which direction boxes expand.
3. Where inter-class boundaries and overlaps form.

```
Order 1: [0.10, A], [0.35, A], [0.42, Test], [0.60, A]   ==> Box [0.10, 0.35] forms, point 0.60 isolated
Order 2: [0.60, A], [0.35, A], [0.42, Test], [0.10, A]   ==> Box [0.35, 0.60] forms, point 0.10 isolated
===> Test point at 0.42 flips classification based solely on presentation sequence!
```

#### Order Ensemble (RQ2 Trust Telemetry)
We exploit this order sensitivity as an asset rather than a defect. By training an ensemble of 5 CFMNN models on identical training data presented in different random permutations (`trust_baselines.order_ensemble`), prediction disagreement across the order ensemble serves as one of the strongest error detection signals in the entire literature (`results/error_detection_v2/`).

---

### 3.2 Computational & Space Complexity Analysis

Let $N$ denote the number of training instances, $d$ the number of feature dimensions, and $H$ the total number of hyperboxes formed ($H \le N$).

#### Time Complexity
1. **Per-Point Box Comparison:** Evaluating membership across existing boxes of class $y$ takes $\mathcal{O}(H_c \cdot d)$, where $H_c$ is the number of boxes belonging to class $y$ ($H_c < H$).
2. **Expansion Check:** Checking the budget condition across candidate boxes takes $\mathcal{O}(H_c \cdot d)$.
3. **Updating Bounds & Centroids:** Updating scalar sums takes $\mathcal{O}(d)$.
4. **Overall Training Time Complexity:**
   $$\mathcal{T}_{\text{train}} = \mathcal{O}\big(N \cdot H \cdot d\big)$$
   In practice, because $H_c \ll N$ on structured datasets (e.g., $H = 26$ on Landsat, $H = 53$ on MAGIC), training executes in fractions of a second ($0.04$ s on Wine, $0.18$ s on Landsat, $1.9$ s on Letter).

#### Space Complexity
The memory footprint is strictly governed by hyperbox coordinate storage:
$$\mathcal{S}_{\text{model}} = \mathcal{O}\big(H \cdot d\big)$$
For $H = 1,000$ boxes and $d = 20$ features, model storage is less than $1$ MB, enabling high-density edge deployments.

---

## 4. Cross-References

- For the architectural breakdown of the CFMNN classifier: [`07_Component_By_Component_Architectural_Design.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/07_Component_By_Component_Architectural_Design.md)
- For the file walkthrough of `cfmnn_core.py`: [`08_Codebase_Implementation_Deep_Dive_Core_Modules.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/08_Codebase_Implementation_Deep_Dive_Core_Modules.md)
- For hyperparameter tuning and sensitivity of $\theta$ and $\gamma$: [`11_Hyperparameter_Configuration_and_Sensitivity_Analysis.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/11_Hyperparameter_Configuration_and_Sensitivity_Analysis.md)
- For empirical training wall-clock times and convergence rates: [`15_Component_Wise_Training_Time_and_Convergence_Analysis.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/15_Component_Wise_Training_Time_and_Convergence_Analysis.md)
