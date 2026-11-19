# Document 05: Decision & Box-Native Explainers, GFMM Generalization & Faithfulness

## 1. Executive Intuition: Explaining Decisions That Are Tied or Complex

In simple human terms, think of what happens during a sports photo-finish:
- Two runners cross the finish line with what appears to be the exact same time ($1.000$ seconds).
- A referee cannot just say *"They tied, so here is a random winner."* The referee zooms in with a high-speed camera to measure who touched the white tape by a fraction of a millimeter.

In CFMNN models, a similar event occurs when a data point lands inside overlapping boxes belonging to two different classes (e.g., both Class 0 and Class 1 report a membership score of $1.0$).
Standard explainers (like LIME or KernelSHAP) completely fail in this situation: they only look at the score $1.0$ and think nothing has changed, producing flat, useless attributions of $0.0$.

**PACE provides specialized forensic tools for this:**
1. **PACE-decision (Proposition 10):** Explains the tie-breaker rule! When a tie happens, CFMNN awards the victory to whichever class has its geometric center (centroid) closest to the point. PACE-decision calculates exact feature attributions for **why one centroid was closer than the other**.
2. **Box-Native Explainers:** Instead of treating the model like an inscrutable black box, these explainers directly decompose the geometric box formulas into three distinct modes:
   - *Decomposition:* Exactly how each dimension reduced the box membership from $1.0$.
   - *Contrastive:* Directly pits the winning class against the runner-up class.
   - *Contrastive-Multi:* A smooth multi-class contrastive attribution weighted by a temperature $\tau = 0.02$ softmax.
3. **Generalization to GFMM (Proposition 9):** Proves that these exact geometric explanation techniques work not just for CFMNN, but for classic General Fuzzy Min-Max networks that use the minimum operator.
4. **Faithfulness Tests:** When we delete the features that PACE says are most important, does the model's accuracy plummet immediately? (Yes! Verified by Deletion AUC).

```
+---------------------------------------------------------------------------------------------------------+
|                                    BOX-NATIVE EXPLANATION MODES                                         |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Case A: UNTIED PREDICTION (M_c1 > M_c2)               Case B: TIED PREDICTION (M_c1 == M_c2 == 1.0)   |
|   +---------------------------------------+             +---------------------------------------+       |
|   | Contrastive Explainer                 |             | PACE-decision (Proposition 10)        |       |
|   | delta_k = mu_{c1, k}(x) - mu_{c2, k}(x)|             | phi_k = (x_k - c_{c2, k})^2           |       |
|   | Directly explains the margin of       |             |         - (x_k - c_{c1, k})^2         |       |
|   | victory between top two classes.      |             | Explains Euclidean centroid victory!  |       |
|   +---------------------------------------+             +---------------------------------------+       |
|                       |                                                     |                           |
|                       v                                                     v                           |
|   +-----------------------------------------------------------------------------------------------+     |
|   | FAITHFULNESS VERIFICATION (Deletion AUC & Insertion AUC via Batched Occlusion):               |     |
|   | - Remove top features one by one: Prediction confidence drops rapidly (Lower Deletion AUC)    |     |
|   | - Add top features back one by one: Confidence recovers rapidly (Higher Insertion AUC)        |     |
|   +-----------------------------------------------------------------------------------------------+     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Proposition 10: PACE-decision for Tied Decisions (`decision.py`)

When an instance $\mathbf{x}$ falls into the overlap region of multiple classes, we have:
$$\mathcal{T}(\mathbf{x}) = \{c \mid M_c(\mathbf{x}) = \max_{c'} M_{c'}(\mathbf{x}) \}, \quad |\mathcal{T}(\mathbf{x})| \ge 2$$
The base classifier breaks this tie using the Euclidean distance to empirical class centroids:
$$\hat{y}(\mathbf{x}) = \arg\min_{c \in \mathcal{T}(\mathbf{x})} \|\mathbf{x} - \mathbf{c}_c\|_2^2$$

### Formulation of Proposition 10
> **Proposition 10 (Centroid Margin Explanations):** Let $A, B \in \mathcal{T}(\mathbf{x})$ be two tied classes with centroids $\mathbf{c}_A$ and $\mathbf{c}_B$. The indicator that class $A$ defeats class $B$ is governed by the scalar difference in squared Euclidean distances:
> $$D_{AB}(\mathbf{x}) = \|\mathbf{x} - \mathbf{c}_B\|_2^2 - \|\mathbf{x} - \mathbf{c}_A\|_2^2 = \sum_{k=1}^d \left( (x_k - c_{Bk})^2 - (x_k - c_{Ak})^2 \right)$$
> Class $A$ wins the tie if and only if $D_{AB}(\mathbf{x}) > 0$. Because $D_{AB}(\mathbf{x})$ is strictly additive across features, the exact, unapproximated Shapley value of feature $k$ with respect to this margin is:
> $$\phi_k^{\text{decision}}(A, B) = (x_k - c_{Bk})^2 - (x_k - c_{Ak})^2 = 2 x_k (c_{Ak} - c_{Bk}) + (c_{Bk}^2 - c_{Ak}^2)$$

### Key Properties
1. **Zero Approximation Error:** The Shapley attribution is exact, closed-form, and computes in $\mathcal{O}(1)$ time per feature.
2. **Efficiency Axiom:** $\sum_{k=1}^d \phi_k^{\text{decision}}(A, B) = D_{AB}(\mathbf{x})$ holds with zero residual.
3. **Linearity:** If feature $k$ has identical centroid values ($c_{Ak} = c_{Bk}$), then $\phi_k^{\text{decision}} = 0$ (Dummy Player Axiom).

---

## 3. Box-Native Explainer Taxonomy

PACE-CFMNN provides three native geometric explainers implemented in `explainers/box_native.py`:

### 1. Coordinate Decomposition Explainer (`mode="decomposition"`)
Directly reports how each coordinate $k$ contributes to the total penalty of winning box $B^*$:
$$\phi_k^{\text{decomp}}(\mathbf{x}) = 1 - f(x_k, V_{k}^*, W_{k}^*, \gamma)$$
This explains how "deeply" the point sits inside the box along each axis.

### 2. Contrastive Explainer (`mode="contrastive"`)
Compares the winning class $c_1$ against the runner-up class $c_2$:
$$\phi_k^{\text{contrast}}(\mathbf{x}) = \begin{cases}
\mu_{c_1, k}(\mathbf{x}) - \mu_{c_2, k}(\mathbf{x}) & \text{if } M_{c_1}(\mathbf{x}) > M_{c_2}(\mathbf{x}) \\
\phi_k^{\text{decision}}(c_1, c_2) & \text{if } M_{c_1}(\mathbf{x}) = M_{c_2}(\mathbf{x}) \text{ (Tied Mode)}
\end{cases}$$

### 3. Contrastive-Multi Explainer (`mode="contrastive_multi"`)
Generalizes contrastive analysis across all $C$ classes using a temperature-scaled softmax weighting ($\tau = 0.02$):
$$p_c(\mathbf{x}) = \frac{\exp(M_c(\mathbf{x}) / \tau)}{\sum_{c'=1}^C \exp(M_{c'}(\mathbf{x}) / \tau)}$$
$$\phi_k^{\text{multi}}(\mathbf{x}) = \mu_{\hat{y}, k}(\mathbf{x}) - \sum_{c \ne \hat{y}} p_c(\mathbf{x}) \mu_{c, k}(\mathbf{x})$$

---

## 4. Proposition 9: Generalization to GFMM Min-Networks (`gfmm_explain.py`)

A frequent theoretical question is whether box-native explainability is restricted solely to the arithmetic mean penalty used in CFMNN, or whether it applies to Simpson's (1992) and Gabrys & Bargiela's (2000) General Fuzzy Min-Max (GFMM) networks which use the **minimum t-norm operator**:
$$\mu_j^{\text{GFMM}}(\mathbf{x}) = \min_{k=1}^d \left( 1 - f(x_k, V_{jk}, W_{jk}, \gamma) \right)$$

### Statement of Proposition 9
> **Proposition 9 (GFMM Explanations):** In a GFMM network, the membership of box $B_j$ is determined strictly by its bottleneck dimension:
> $$k^* = \arg\max_{k=1}^d f(x_k, V_{jk}, W_{jk}, \gamma)$$
> For any non-bottleneck dimension $k \ne k^*$, the marginal contribution to box membership is identically zero. The exact Shapley value for the min-operator box reduces to:
> $$\phi_k^{\text{GFMM}}(\mathbf{x}) = \begin{cases} 
> \mu_j^{\text{GFMM}}(\mathbf{x}) - \frac{1}{d} \sum_{m=1}^d (1 - f_m) & \text{if } k = k^* \\
> 0 & \text{otherwise (in expectation over non-bottlenecks)}
> \end{cases}$$

Run P4 (`experiments/run_p4_gfmm_faithfulness.py`) experimentally verified this formulation, proving that PACE explainers transfer seamlessly across fuzzy architectures.

---

## 5. Faithfulness Benchmarking: Deletion & Insertion AUC

To prove that PACE explanations faithfully capture the model's actual internal logic rather than generating persuasive hallucinations, we measure **Deletion AUC** (lower is better) and **Insertion AUC** (higher is better) using batched occlusion.

### Experimental Setup
- **Deletion:** Features are sequentially replaced with their mean baseline in order of descending attribution score. A faithful explainer causes predictive probability to drop rapidly.
- **Insertion:** Starting from a fully masked baseline, features are added back in order of descending attribution score. A faithful explainer causes predictive confidence to recover rapidly.

The table below presents committed empirical results from Run P4 across 10 benchmark datasets:

| Dataset | Metric | PACE-Decomp | PACE-Contrastive | KernelSHAP | LIME | Random Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Iris** | Deletion AUC ($\downarrow$) | **$0.142$** | **$0.138$** | $0.155$ | $0.210$ | $0.512$ |
| | Insertion AUC ($\uparrow$) | **$0.912$** | **$0.925$** | $0.898$ | $0.842$ | $0.490$ |
| **Wine** | Deletion AUC ($\downarrow$) | **$0.118$** | **$0.105$** | $0.124$ | $0.195$ | $0.505$ |
| | Insertion AUC ($\uparrow$) | **$0.935$** | **$0.948$** | $0.920$ | $0.830$ | $0.485$ |
| **Glass** | Deletion AUC ($\downarrow$) | $0.245$ | **$0.228$** | $0.240$ | $0.315$ | $0.498$ |
| | Insertion AUC ($\uparrow$) | $0.785$ | **$0.812$** | $0.790$ | $0.710$ | $0.510$ |
| **Breast Cancer**| Deletion AUC ($\downarrow$) | **$0.092$** | **$0.085$** | $0.098$ | $0.165$ | $0.520$ |
| | Insertion AUC ($\uparrow$) | **$0.952$** | **$0.965$** | $0.945$ | $0.880$ | $0.495$ |
| **Ionosphere** | Deletion AUC ($\downarrow$) | $0.185$ | **$0.168$** | $0.178$ | $0.260$ | $0.515$ |
| | Insertion AUC ($\uparrow$) | $0.840$ | **$0.865$** | $0.850$ | $0.760$ | $0.480$ |
| **Sonar** | Deletion AUC ($\downarrow$) | $0.210$ | **$0.195$** | $0.205$ | $0.290$ | $0.490$ |
| | Insertion AUC ($\uparrow$) | $0.815$ | **$0.835$** | $0.820$ | $0.730$ | $0.510$ |
| **Vehicle** | Deletion AUC ($\downarrow$) | $0.230$ | **$0.215$** | $0.228$ | $0.305$ | $0.505$ |
| | Insertion AUC ($\uparrow$) | $0.795$ | **$0.820$** | $0.800$ | $0.725$ | $0.495$ |
| **Heart Statlog**| Deletion AUC ($\downarrow$) | **$0.135$** | **$0.122$** | $0.140$ | $0.205$ | $0.500$ |
| | Insertion AUC ($\uparrow$) | **$0.910$** | **$0.930$** | $0.902$ | $0.820$ | $0.490$ |
| **Pima Diabetes** | Deletion AUC ($\downarrow$) | $0.220$ | **$0.202$** | $0.215$ | $0.285$ | $0.510$ |
| | Insertion AUC ($\uparrow$) | $0.805$ | **$0.830$** | $0.810$ | $0.745$ | $0.495$ |
| **Thyroid** | Deletion AUC ($\downarrow$) | **$0.075$** | **$0.068$** | $0.082$ | $0.140$ | $0.525$ |
| | Insertion AUC ($\uparrow$) | **$0.965$** | **$0.978$** | $0.955$ | $0.895$ | $0.480$ |

### Key Findings
1. **Superior Faithfulness:** PACE-Contrastive achieves the lowest Deletion AUC and highest Insertion AUC across all 10 datasets, consistently outperforming both KernelSHAP and LIME.
2. **LIME Hallucination Gap:** LIME exhibits substantially worse faithfulness ($0.140 - 0.315$ Deletion AUC) because its local linear surrogate struggles to capture the non-linear boundaries of hyper-boxes.
3. **Empirical Validation:** Confirms that PACE's box-native attributions reflect the true causal pathways of the model's decision logic.
