# 06. Mathematical Foundations and Theoretical Formulation

## 1. Mathematical Notation & Geometric Preliminaries

All data points reside within a $d$-dimensional unit hypercube: $x = (x_1, \dots, x_d) \in [0, 1]^d$.
A trained Centroid-Based Fuzzy Min-Max Neural Network (CFMNN) is parameterized by a finite set of $H$ hyperboxes:
$$\mathcal{B} = \{ B_j \}_{j=1}^H$$
Each hyperbox $B_j$ is defined by the 4-tuple:
$$B_j = \big( v_j, \, w_j, \, c_j, \, \mu_j \big)$$
where:
* $v_j = (v_{j1}, \dots, v_{jd}) \in [0, 1]^d$ is the minimum coordinate corner.
* $w_j = (w_{j1}, \dots, w_{jd}) \in [0, 1]^d$ is the maximum coordinate corner, with the geometric constraint $v_{ji} \le w_{ji}$ for all $i \in \{1, \dots, d\}$.
* $c_j \in \{0, \dots, C-1\}$ is the categorical class label assigned to box $j$.
* $\mu_j = (\mu_{j1}, \dots, \mu_{jd}) \in [0, 1]^d$ is the internal centroid vector, representing the empirical center of mass of all training points absorbed by box $j$.

---

## 2. Core CFMNN Formulation

### 2.1 The Simpson Ramp Function
The activation profile of a box face is determined by Simpson's continuous ramp function $f: \mathbb{R} \to [0, 1]$, parameterized by sensitivity slope $\gamma > 0$ (default $\gamma = 4.0$):
$$f(t) = \max\Big(0, \; 1 - \max\big(0, \; \gamma \cdot \min(1, t)\big)\Big)$$

#### Analytical Piecewise Behavior
$$f(t) = \begin{cases} 
1.0 & \text{if } t \le 0 \\
1 - \gamma t & \text{if } 0 < t < \frac{1}{\gamma} \\
0.0 & \text{if } t \ge \frac{1}{\gamma}
\end{cases}$$
For $\gamma = 4.0$, $f(t)$ is $1.0$ for non-positive arguments, drops linearly with slope $-4.0$ between $0.0$ and $0.25$, and remains identically $0.0$ for all $t \ge 0.25$.

---

### 2.2 Univariate Feature Terms
For a query coordinate $x_i \in [0, 1]$ and hyperbox bounds $[v_{ji}, w_{ji}]$, the per-feature membership term $m_{ji}(x_i)$ evaluates the proximity of $x_i$ to the interval:
$$m_{ji}(x_i) = \frac{1}{2} \Big[ f(x_i - w_{ji}) + f(v_{ji} - x_i) \Big]$$

#### Mathematical Properties of $m_{ji}(x_i)$
1. **Internal Invariance:** If $x_i \in [v_{ji}, w_{ji}]$, then $x_i - w_{ji} \le 0$ and $v_{ji} - x_i \le 0$. Hence $f(x_i - w_{ji}) = 1$ and $f(v_{ji} - x_i) = 1$, yielding:
   $$m_{ji}(x_i) = \frac{1}{2} [1 + 1] = 1.0$$
2. **External Decay:** Because $v_{ji} \le w_{ji}$, an input $x_i$ can violate at most one boundary face at a time. If $x_i > w_{ji}$, then $v_{ji} - x_i < 0 \implies f(v_{ji} - x_i) = 1.0$. The term becomes:
   $$m_{ji}(x_i) = \frac{1}{2} \Big[ f(x_i - w_{ji}) + 1.0 \Big] = 1 - \frac{\gamma}{2} (x_i - w_{ji})$$
3. **Bounded Range:** For any coordinate $x_i$, $m_{ji}(x_i) \in [0.5, 1.0]$. The lower floor of $0.5$ is reached whenever the distance exceeds $1/\gamma = 0.25$.

---

### 2.3 Total Hyperbox Membership
Total membership of input vector $x$ in hyperbox $j$ is the unweighted arithmetic mean across all $d$ dimensions:
$$b_j(x) = \frac{1}{d} \sum_{i=1}^d m_{ji}(x_i)$$
* Range: $b_j(x) \in [0.5, 1.0]$.
* Bounding Condition: $b_j(x) = 1.0 \iff \forall i \in \{1, \dots, d\}, \; v_{ji} \le x_i \le w_{ji} \iff x \in B_j$.

---

### 2.4 Class Supports and Margin Formulation
For a target class $c \in \{0, \dots, C-1\}$, its raw support $s_c(x)$ is the maximum membership among all hyperboxes assigned to class $c$:
$$s_c(x) = \max_{j : c_j = c} b_j(x)$$
If class $c$ contains no hyperboxes, $s_c(x) \triangleq 0.5$.
The affine rescaled support $s'_c(x) \in [0, 1]$ maps the $[0.5, 1.0]$ membership domain onto the standard unit interval:
$$s'_c(x) = 2 s_c(x) - 1$$
For predicted class $A$, its prediction margin $g_A(x)$ over competitor classes is:
$$g_A(x) = s'_A(x) - \max_{c \ne A} s'_c(x) = 2 \left[ s_A(x) - \max_{c \ne A} s_c(x) \right]$$

---

### 2.5 Inference, Centroid Tie-Breaking, and Winner Selection
Let the maximum membership across the entire network be:
$$b^*(x) = \max_{j \in \{1, \dots, H\}} b_j(x)$$
Given numerical tie-tolerance $\tau = 10^{-9}$, the tied top group $T(x)$ is defined as:
$$T(x) = \big\{ j \in \{1, \dots, H\} : b_j(x) \ge b^*(x) - \tau \big\}$$
For each candidate box $j \in T(x)$, compute the squared Euclidean distance to its centroid:
$$D_j(x) = \|x - \mu_j\|_2^2 = \sum_{i=1}^d (x_i - \mu_{ji})^2$$
The winning hyperbox $j^*(x)$ is uniquely determined by:
$$j^*(x) = \arg\min_{j \in T(x)} D_j(x)$$
Any remaining tie in $D_j(x)$ is broken deterministically by selecting the lowest box index $j$.
The final predicted class is:
$$\hat{y}(x) = c_{j^*(x)}$$

---

## 3. Formal Theoretical Propositions & Proofs

### Proposition 1: Every CFMNN Training Error is an Overlapping Cross-Class Tie
* **Statement:** Let $M$ be a CFMNN trained on training set $\mathcal{D}_{\text{train}} = \{(x_r, y_r)\}_{r=1}^N$ via single-pass expansion under any size limit $\theta \in (0, 1]$, any sensitivity $\gamma > 0$, any row presentation order, and under either centroid rule (`"fixed"` or `"original"`) and either expansion rule (`"search"` or `"best"`). Then:
  1. For every training instance $(x_r, y_r)$, there exists at least one box $k$ such that $c_k = y_r$ and $x_r \in B_k$, implying $\max_{j : c_j = y_r} b_j(x_r) = 1.0$.
  2. If $M$ misclassifies $x_r$ ($\hat{y}(x_r) \ne y_r$), then the winning hyperbox $j^*(x_r)$ has class $c_{j^*} \ne y_r$ and contains $x_r$ ($b_{j^*}(x_r) \ge 1 - \tau$).
  3. Consequently, every training error is an exact cross-class tie where $b_{j^*}(x_r) = b_k(x_r) = 1.0$, which the nearest-centroid tie-breaker resolved in favor of the incorrect class.

* **Proof:**
  1. Consider training step $r$ when instance $(x_r, y_r)$ is presented. Exactly one of three actions occurs:
     - Case A: $x_r$ already lies inside an existing box of class $y_r$.
     - Case B: An existing box of class $y_r$ is expanded to absorb $x_r$ such that $\sum_i (w_{ji} - v_{ji}) \le d \cdot \theta$.
     - Case C: A new point box is created with $v = w = x_r$.
     In all three cases, immediately after step $r$, $x_r$ lies inside a box $k$ with $c_k = y_r$. Subsequent training steps only expand existing boxes ($V \le V_{\text{new}}, W \ge W_{\text{new}}$) or create new boxes; boxes are never shrunk or deleted. Therefore, $x_r \in B_k$ holds permanently at the conclusion of training. Inside $B_k$, every feature term $m_{ki}(x_{ri}) = 1.0$, so $b_k(x_r) = \frac{1}{d} \sum_{i=1}^d 1.0 = 1.0$.
  2. Because $b_k(x_r) = 1.0$, the global maximum membership $b^*(x_r) = 1.0$. By definition, any candidate in the tied top group satisfies $b_j(x_r) \ge 1.0 - \tau$. If the winning box $j^*$ predicts class $c_{j^*} \ne y_r$, it must belong to $T(x_r)$, and for $\tau = 0$, $b_{j^*}(x_r) = 1.0$, which geometrically requires $x_r \in B_{j^*}$. Both boxes achieve maximum membership $1.0$, and the winner is chosen solely by $\arg\min D_j(x_r)$. $\blacksquare$

---

### Proposition 2: Additive Coalition Values Without Model Evaluation
* **Statement:** Take query point $x \in [0, 1]^d$, background instance $z \in [0, 1]^d$, and subset of features $S \subseteq \{1, \dots, d\}$. Define the hybrid coalitional point $h_S \in [0, 1]^d$ by:
  $$h_{S, i} = \begin{cases} x_i & \text{if } i \in S \\ z_i & \text{if } i \notin S \end{cases}$$
  Then for any hyperbox $B_j$, its membership at $h_S$ is exactly:
  $$b_j(h_S) = b_j(z) + \sum_{i \in S} \Delta_{ji}(x_i, z_i)$$
  where the additive delta term is:
  $$\Delta_{ji}(x_i, z_i) = \frac{m_{ji}(x_i) - m_{ji}(z_i)}{d}$$

* **Proof:**
  By definition, box membership is an unweighted sum of decoupled univariate terms:
  $$b_j(h_S) = \frac{1}{d} \sum_{i=1}^d m_{ji}(h_{S, i}) = \frac{1}{d} \sum_{i \in S} m_{ji}(x_i) + \frac{1}{d} \sum_{i \notin S} m_{ji}(z_i)$$
  Expressing the second summation over all dimensions minus $S$:
  $$\frac{1}{d} \sum_{i \notin S} m_{ji}(z_i) = \frac{1}{d} \sum_{i=1}^d m_{ji}(z_i) - \frac{1}{d} \sum_{i \in S} m_{ji}(z_i) = b_j(z) - \frac{1}{d} \sum_{i \in S} m_{ji}(z_i)$$
  Substituting back yields:
  $$b_j(h_S) = b_j(z) + \sum_{i \in S} \frac{m_{ji}(x_i) - m_{ji}(z_i)}{d} = b_j(z) + \sum_{i \in S} \Delta_{ji}(x_i, z_i) \quad \blacksquare$$

---

### Proposition 3: Exact Bounding Box Pruning Invariants
* **Statement:** Fix query point $x$, background row $z$, and partition hyperboxes into two disjoint sets: $\mathcal{B}_A = \{j : c_j = A\}$ and $\mathcal{B}_{\text{other}} = \{j : c_j \ne A\}$.
  For each box $j$, define its analytical coalitional floor and ceiling:
  $$\text{floor}_j = b_j(z) + \sum_{i=1}^d \min\big(0, \, \Delta_{ji}\big), \qquad \text{ceil}_j = b_j(z) + \sum_{i=1}^d \max\big(0, \, \Delta_{ji}\big)$$
  Then box $j \in \mathcal{G}$ (where $\mathcal{G} \in \{\mathcal{B}_A, \mathcal{B}_{\text{other}}\}$) can be permanently removed from evaluation without changing the margin $g_A(h_S)$ for any coalition $S \subseteq \{1, \dots, d\}$ if:
  $$\text{ceil}_j < \max_{k \in \mathcal{G}} \text{floor}_k$$

* **Proof:**
  For any coalition $S$, since $\Delta_{ji}$ is added if $i \in S$ and omitted otherwise:
  $$\sum_{i=1}^d \min(0, \Delta_{ji}) \le \sum_{i \in S} \Delta_{ji} \le \sum_{i=1}^d \max(0, \Delta_{ji})$$
  Therefore, $b_j(h_S) \in [\text{floor}_j, \text{ceil}_j]$ for all $2^d$ possible coalitions $S$.
  Let $k^* = \arg\max_{k \in \mathcal{G}} \text{floor}_k$. For all $S$:
  $$b_j(h_S) \le \text{ceil}_j < \text{floor}_{k^*} \le b_{k^*}(h_S) \le \max_{k \in \mathcal{G}} b_k(h_S)$$
  Because $b_j(h_S)$ is strictly strictly dominated by $b_{k^*}(h_S)$ under every coalition, box $j$ can never achieve the group maximum, and its elimination does not alter $\max_{k \in \mathcal{G}} b_k(h_S)$ or margin $g_A(h_S)$. $\blacksquare$

---

### Proposition 4: Formal Prediction Invariance Certificates
* **Statement:** Let $M$ be a trained CFMNN and $M'$ be an edited model obtained by modifying, adding, or removing a set of hyperboxes $\mathcal{C}_{\text{changed}} \subset \mathcal{B}$.
  Let $T(x)$ and $T'(x)$ denote the tied top groups of query point $x$ under $M$ and $M'$, respectively.
  If:
  $$\big( T(x) \cap \mathcal{C}_{\text{changed}} = \emptyset \big) \quad \land \quad \big( T'(x) \cap \mathcal{C}_{\text{changed}} = \emptyset \big)$$
  then the predicted class is guaranteed to be invariant:
  $$\hat{y}(x; M') \equiv \hat{y}(x; M)$$

* **Proof:**
  If $T(x) \cap \mathcal{C}_{\text{changed}} = \emptyset$, then all hyperboxes achieving the maximum membership under $M$ are completely unchanged.
  Similarly, if $T'(x) \cap \mathcal{C}_{\text{changed}} = \emptyset$, the top group under $M'$ contains only unchanged hyperboxes.
  Because unchanged hyperboxes have identical bounds $V, W$, their membership values $b_j(x)$ are identical in $M$ and $M'$.
  Therefore, the maximum membership values are identical: $\max_k b_k(x; M) = \max_k b_k(x; M')$, which implies $T(x) = T'(x)$.
  Since the centroids $\mu_j$ and box classes $c_j$ of all boxes in $T(x)$ are identical across both models, the nearest-centroid tie-breaker evaluates identical distances $D_j(x)$ and picks the exact same winning box $j^*$, guaranteeing $\hat{y}(x; M') = \hat{y}(x; M)$. $\blacksquare$

---

### Proposition 5: Exact PACE-SHAP Computation is #P-Hard
* **Statement:** Given a CFMNN with 2 classes and 4 hyperboxes, query point $x$, and single background row $z$, computing the exact PACE-SHAP value vector $\phi \in \mathbb{Q}^d$ is #P-hard with respect to binary input length.

* **Proof by Polynomial Reduction from Weighted Voting Games:**
  1. **Source Problem:** In a weighted voting game, $n$ players have positive integer weights $w_1, \dots, w_n$ and quota $q \in [1, \sum w_i]$. A coalition $S \subseteq \{1, \dots, n\}$ wins if $\sum_{i \in S} w_i \ge q$. Player $i$'s Shapley-Shubik power index $SS_i$ measures the fraction of permutations where player $i$ turns a losing coalition into a winning one. Computing $SS_i$ is #P-complete under binary weight encoding (Deng & Papadimitriou 1994, Faliszewski & Hemaspaandra 2009).
  2. **Gadget Construction (`hardness.voting_game_model`):**
     Construct $d = 2n$ features: $n$ weight dimensions followed by $n$ padding dimensions.
     Set $\gamma = 4.0$, $r = 1/\gamma = 0.25$, and $\epsilon = \frac{1}{4 n \max_i w_i}$.
     Query point $x = (0.5, \dots, 0.5)$. Background point $z = x$, except on weight features $i \in \{1, \dots, n\}$, set:
     $$z_i = 0.5 - t_i, \quad t_i = \frac{2 d \epsilon w_i}{\gamma}$$
     Construct 4 boxes: $a_1, a_2$ of class $A$, and $b_1, b_2$ of class $B$:
     - On weight features, $a_1$ and $b_1$ span $[0.5, 0.5 + r]$, while $a_2$ and $b_2$ span $[0, 1]$.
     - By Proposition 2, when feature $i$ takes $x_i$, $a_1$ and $b_1$ gain exactly $\Delta_i = \epsilon w_i$, while $a_2$ and $b_2$ gain $0$.
     - On padding features, boxes start at offset $0.5 + s_j$ such that their base memberships at $z$ evaluate to:
       $$b_{a2}(z) = b_{b2}(z) = K = 0.75$$
       $$b_{a1}(z) = K - \epsilon(q - 1), \qquad b_{b1}(z) = K - \epsilon q$$
  3. **Coalition Evaluation:** For any subset $S \subseteq \{1, \dots, n\}$, the class supports evaluate to:
     $$s_A(S) = \max\Big(K, \; K + \epsilon\big(\sum_{i \in S} w_i - q + 1\big)\Big)$$
     $$s_B(S) = \max\Big(K, \; K + \epsilon\big(\sum_{i \in S} w_i - q\big)\Big)$$
     Because $\sum_{i \in S} w_i$ is an integer, the difference $s_A(S) - s_B(S)$ evaluates to $\epsilon$ if $\sum_{i \in S} w_i \ge q$, and $0$ otherwise.
  4. **Attribution Equivalence:**
     The margin is $g_A(h_S) = 2 \epsilon \cdot \mathbb{I}\big(w(S) \ge q\big)$.
     Therefore, the Shapley value of weight feature $i$ is:
     $$\phi_i = 2 \epsilon \cdot SS_i$$
     A polynomial-time algorithm for exact PACE-SHAP would compute $SS_i$ in polynomial time, proving #P-hardness. $\blacksquare$

---

### Proposition 6: Condition S (Training-Safe Shrinks)
* **Statement:** An edit shrinking box $k$ on feature $i$ cannot break any correct training prediction ($\text{FixShare} = 1.0$, Zero Regression) if:
  1. Box $k$ contains zero training members with correct predictions ($|\text{members}(k) \cap \{r : \hat{y}(x_r) = y_r\}| = 0$).
  2. The boundary cut does not drop any training member of box $k$ that relies on $k$ for class support.
  3. No training point of another class had $k$ as a non-tied runner-up whose disappearance flips a tie.

---

### Proposition 7: Explanation Invariance Certificates
* **Statement:** A model edit altering boxes in $\mathcal{C}_{\text{changed}}$ cannot alter the exact PACE-SHAP attribution vector $\phi(x)$ of query instance $x$ if, across all background rows $z_r$ and all coalitions $S \subseteq \{1, \dots, d\}$, no changed box ever achieves the maximum membership of its class:
  $$\forall r \in \{1, \dots, R\}, \; \forall j \in \mathcal{C}_{\text{changed}}, \quad \text{ceil}_{j, r} < \max_{k \in \text{class}(j)} \text{floor}_{k, r}$$

---

### Proposition 8: Exact Analytical Single-Feature Flip Sets
* **Statement:** For query point $x$ and feature $i$, the winning class $\hat{y}(x_i = t, x_{-i})$ as a function of continuous coordinate $t \in [0, 1]$ is a piecewise constant step function with at most $4H + 2$ critical transition breakpoints:
  $$\mathcal{T}_i = \bigcup_{j=1}^H \left\{ v_{ji} - \frac{1}{\gamma}, \, v_{ji}, \, w_{ji}, \, w_{ji} + \frac{1}{\gamma} \right\} \cap [0, 1]$$
  Within each open interval between consecutive sorted breakpoints, all membership terms $m_{ji}(t)$ are purely linear affine functions of $t$. Class intersections and centroid tie-breaks within each interval can be solved analytically in closed form without grid discretization.

---

### Proposition 9: Generalization to Min-Membership Networks (GFMM)
* **Statement:** In General Fuzzy Min-Max networks (Simpson 1992, Gabrys & Bargiela 2000), box membership uses the min-operator: $b_j(x) = \min_{i=1}^d m_{ji}(x_i)$.
  While $b_j$ is no longer additive, the coalitional value across subset $S$ can be evaluated by:
  $$b_j(h_S) = \min\left( \min_{i \in S} m_{ji}(x_i), \; \min_{i \notin S} m_{ji}(z_i) \right)$$
  By sorting univariate feature terms once, exact PACE-SHAP for GFMM (`PaceShapMin`) evaluates in $O(H \cdot d \log d + H \cdot 2^d)$ without model queries.

---

### Proposition 10: Exact Decision-Level Shapley Values (PACE-decision)
* **Statement:** Let the categorical prediction function be an indicator $f(x) = \mathbb{I}(\hat{y}(x) = A)$. Unlike the margin $g_A(x)$, $f(x)$ incorporates the nearest-centroid tie-breaker. By integrating Proposition 8's closed-form decision boundaries, PACE-decision evaluates the exact Shapley value of the categorical decision:
  $$\phi_i^{\text{dec}} = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (d - |S| - 1)!}{d!} \Big[ \mathbb{I}\big(\hat{y}(x_{S \cup \{i\}}, z_{-S \setminus \{i\}}) = A\big) - \mathbb{I}\big(\hat{y}(x_S, z_{-S}) = A\big) \Big]$$

---

## 4. Complexity & Dynamic Programming Recurrence Engine

In standard exact PACE-SHAP (`exact_algo="matmul"`), computing subset memberships requires:
$$M(S) = b(z) + \Delta \cdot \mathbf{1}_S$$
Evaluating all $2^d$ coalitions across $K$ non-pruned boxes requires:
$$\mathcal{O}\big(K \cdot d \cdot 2^d\big) \text{ operations}$$

### The Dynamic Recurrence Speedup (`exact_algo="recurrence"`)
By recognizing that subset sums exhibit optimal substructure:
$$S = S' \cup \{i\} \implies \sum_{k \in S} \Delta_k = \sum_{k \in S'} \Delta_k + \Delta_i$$
The dynamic recurrence engine orders subsets by gray code / bitmask succession, requiring only **1 addition per coalition**:
$$\mathcal{O}\big(K \cdot 2^d\big) \text{ additions}$$
This eliminates the factor of $d$ multiplications, accelerating exact evaluation by **$1.03\times$ to $2.13\times$** in empirical benchmarks (`results/recurrence/summary.md`).

---

## 5. Cross-References

- For the component-level architectural implementation: [`07_Component_By_Component_Architectural_Design.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/07_Component_By_Component_Architectural_Design.md)
- For the file-by-file code walkthrough of `cfmnn_core.py`, `explain.py`, and `hardness.py`: [`08_Codebase_Implementation_Deep_Dive_Core_Modules.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/08_Codebase_Implementation_Deep_Dive_Core_Modules.md)
- For the formal proofs and verification tests of Propositions 5–10: [`21_Tough_Review_Audit_and_Formal_Proof_Extensions.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/21_Tough_Review_Audit_and_Formal_Proof_Extensions.md)
- For the complete analysis of Proposition 1 and Proposition 4 in self-repair: [`22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md)
