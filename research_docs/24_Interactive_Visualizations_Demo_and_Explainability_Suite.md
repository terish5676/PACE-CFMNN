# 24. Interactive Visualizations, Demo, and Explainability Suite

## 1. Overview of Visual Explainability and Interactive Tools

To translate complex hyperbox geometry and multi-dimensional game-theoretic attributions into actionable insights for machine learning practitioners and end-users, the repository includes a suite of visualization engines and interactive tools:
1. **The Interactive 2-D Web Visualizer (`release/demo/pace_cfmnn_demo.html`):** A standalone client-side JavaScript demonstration engine for real-time training, self-repair, certificate shading, and exact PACE-SHAP waterfall charts.
2. **Contrastive Explanation Engines (`contrastive`, `contrastive_multi`):** Decomposing the relative decision boundary between winning and runner-up classes.
3. **Shapley Interaction Indices (`PaceShap.interactions`):** Quantifying pairwise non-linear feature coupling beyond individual main effects.
4. **Global Feature Importance Aggregators:** Bridging localized instance explanations with global model auditing.

```
+----------------------------------------------------------------------------------------------------+
|                                    VISUAL EXPLAINABILITY ECOSYSTEM                                 |
|                                                                                                    |
|    [ Local Explanations ]             [ Contrastive Logic ]           [ Global Model Auditing ]    |
|    - Exact PACE-SHAP                  - Membership Mode               - Data-Based Mean |phi|      |
|    - PACE-flip Occlusion              - Centroid Tie Mode             - Normalized Mean            |
|    - Sparse Counterfactuals           - Multi-Competitor              - Model-Only Edge Weights    |
|              |                                  |                                 |                |
|              +----------------------------------+---------------------------------+                |
|                                                 |                                                  |
|                                                 v                                                  |
|                      +-----------------------------------------------------+                       |
|                      |      Interactive 2-D Client-Side Browser Visualizer |                       |
|                      |          (release/demo/pace_cfmnn_demo.html)        |                       |
|                      +-----------------------------------------------------+                       |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. The Interactive 2-D Browser Visualizer (`pace_cfmnn_demo.html`)

Located in `release/demo/pace_cfmnn_demo.html`, this single-file interactive application implements a complete, self-contained client-side clone of the CFMNN classifier and PACE explainers in JavaScript/HTML5 Canvas:

```
+----------------------------------------------------------------------------------------------------+
|                                    INTERACTIVE DEMO INTERFACE MAP                                  |
|                                                                                                    |
|    +------------------------------------------+  +--------------------------------------------+    |
|    |      2D Hyperbox Decision Canvas         |  |          Interactive Control Panel         |    |
|    |                                          |  |                                            |    |
|    |     [Class A Box]                        |  |  [Add Point]   Class: ( ) A  (*) B         |    |
|    |     +---------+                          |  |  [Theta Slider]: <-----[ 0.30 ]----->      |    |
|    |     |    * mu |                          |  |  [Train CFMNN]                             |    |
|    |     |  (Overlap)-------+                 |  |  [Run Self-Repair]                         |    |
|    |     +---------|        |                 |  |                                            |    |
|    |               | * mu   |                 |  +--------------------------------------------+    |
|    |               +--------+                 |  |         Live PACE-SHAP Waterfall Chart     |    |
|    |              [Class B Box]               |  |                                            |    |
|    |                                          |  |  Feature x_0: [=========>      ] +0.42     |    |
|    |  Shaded Region: Invariance Certificate   |  |  Feature x_1: [===>            ] +0.18     |    |
|    |  (Certified Zone of Prediction Change)   |  |  Base Value:  [======>         ] +0.30     |    |
|    +------------------------------------------+  +--------------------------------------------+    |
+----------------------------------------------------------------------------------------------------+
```

### Key Interactive Capabilities
1. **Interactive Data Instantiation:** Users click directly on the canvas to place class-labeled training instances in $[0, 1]^2$.
2. **Real-Time CFMNN Fitting:** Renders hyperboxes with semi-transparent fills, minimum/maximum boundary corners, and red dot centroids ($\mu_j$).
3. **Live Decision Surface Shader:** Evaluates membership and nearest-centroid tie-breaking across a dense $200 \times 200$ grid, displaying color-coded decision boundaries.
4. **Interactive Self-Repair & Certificate Shading:**
   - Misclassified points are highlighted with red warning rings.
   - Clicking "Self-Repair" executes `shrink_box()` in real time, moving the boundary face of the winning wrong box past the point.
   - **The Certificate Shade:** Instantly shades the spatial impact set $C_{\text{cert}}$ (Proposition 4), visually demonstrating to the user that predictions outside the shaded zone remain completely unaltered.
5. **Real-Time PACE-SHAP Waterfall Chart:** Clicking any query coordinate on the canvas computes closed-form Shapley values and renders a horizontal waterfall chart illustrating how $x_0$ and $x_1$ move the predicted margin away from the baseline.

---

## 3. Contrastive Explanations: "Why Class A, Not Class B?"

Standard attribution explainers score features relative to an uninformative background mean. In many clinical or legal inquiries, users demand contrastive answers: *"Why was this applicant rejected (Class A) instead of approved (Class B)?"*

Implemented in `pace_cfmnn/explain.py` via `contrastive()` and `contrastive_pair()`:

### 3.1 Dual-Regime Contrastive Mechanics
Given predicted class $A$ and competitor class $B$, the per-feature contrastive vector $e \in \mathbb{R}^d$ operates in two mathematically distinct regimes:

```
+----------------------------------------------------------------------------------------------------+
|                                 CONTRASTIVE EXPLANATION DUAL REGIMES                               |
|                                                                                                    |
|    [ Regime 1: Membership Separation Mode ]                                                        |
|    Condition: Raw memberships differ: |b_A(x) - b_B(x)| > tau                                      |
|    Formula:   e_i = (m_Ai(x_i) - m_Bi(x_i)) / d                                                    |
|    Sum:       sum_i e_i = b_A(x) - b_B(x) > 0                                                      |
|    Meaning:   Positive e_i indicates feature i pushes x deeper into box A than box B.              |
|                                                                                                    |
|    [ Regime 2: Centroid Tie Mode ]                                                                 |
|    Condition: Memberships tie within tolerance: |b_A(x) - b_B(x)| <= tau (Both boxes contain x)    |
|    Formula:   e_i = (x_i - mu_Bi)^2 - (x_i - mu_Ai)^2                                             |
|    Sum:       sum_i e_i = ||x - mu_B||^2 - ||x - mu_A||^2 > 0                                      |
|    Meaning:   Positive e_i indicates feature i pulls x closer to centroid A than centroid B.       |
+----------------------------------------------------------------------------------------------------+
```

### 3.2 Multi-Competitor Contrastive Engine (`contrastive_multi`)
In multi-class settings ($C > 2$), multiple competing boxes may challenge the winner. `contrastive_multi()` ranks the top $k$ competing boxes of other classes and aggregates normalized contrastive vectors weighted by boundary proximity:
$$w_c \propto \exp\left( -\frac{\max(0, \, b_A(x) - b_c(x))}{\tau_{\text{temp}}} \right), \quad \tau_{\text{temp}} = 0.02$$
Boxes on the verge of winning receive the highest weight, providing an ultra-fast ($0.02 - 0.47$ ms) overview of multi-class competition.

---

## 4. Shapley Interaction Values (RQ1 Extension B4)

While standard Shapley values $\phi_i$ allocate main effect attributions to individual features, non-linear models frequently exhibit feature coupling (e.g., medical risks that emerge only when high blood pressure AND high cholesterol occur simultaneously).

Implemented in `PaceShap.interactions()` via closed-form interaction evaluations:

### 4.1 Mathematical Formulation of Interaction Indices
For feature pair $(i, j)$ with $i \ne j$, the Shapley interaction index $I_{ij}$ is defined as:
$$I_{ij} = \sum_{S \subseteq \{1, \dots, d\} \setminus \{i, j\}} \frac{|S|! (d - |S| - 2)!}{2 \cdot (d - 1)!} \Big[ g_A(h_{S \cup \{i, j\}}) - g_A(h_{S \cup \{i\}}) - g_A(h_{S \cup \{j\}}) + g_A(h_S) \Big]$$

#### Closed-Form Evaluation via Proposition 2
Because every coalition margin $g_A(h_S)$ is evaluated in closed form without querying the model, `PaceShap.interactions()` evaluates all $\binom{d}{2}$ interaction pairs simultaneously from the same pre-computed $\Delta$ table.

### 4.2 Empirical Validation on Synthetic Suite S3 (Pre-Registered B4)
Synthetic dataset S3 incorporates a true ground-truth conjunctive AND rule:
$$y = 1 \iff (x_0 \ge 0.5) \land (x_1 \ge 0.5) \land (x_2 \ge 0.5)$$
Evaluated across 10 pre-registered seeds (seeds 40–49; `results/extensions_v1/interactions/runs.csv`):

```
+----------------------------------------------------------------------------------------------------+
|                                    S3 INTERACTION ATTRIBUTION MATRIX                               |
|                                                                                                    |
|    Feature Pair            True Relationship         Interaction Index I_ij     Rank               |
|    --------------------------------------------------------------------------------------------    |
|    (x_0, x_1)              Conjunctive AND           0.245 +/- 0.018            #1 Strongest Pair  |
|    (x_0, x_2)              Conjunctive AND           0.238 +/- 0.015            #2                 |
|    (x_1, x_2)              Conjunctive AND           0.241 +/- 0.016            #3                 |
|    (x_0, x_3)              Noise Interaction         0.012 +/- 0.004            Noise Floor        |
|    (x_3, x_4)              Noise Interaction         0.002 +/- 0.001            Noise Floor        |
+----------------------------------------------------------------------------------------------------+
```
* **Empirical Outcome:** On S3, PACE interaction values identified the true AND pair as the **strongest interaction in 10 of 10 experimental seeds**.
* **Key Academic Distinction:** The interaction indices also identified $(x_0, x_1)$ as strongest on S1 and S2, even though S1's true rule is additive ($x_0^2 + x_1^2 + x_2^2$). This confirms that **Shapley interaction values describe the geometry of the classifier's internal hyperboxes, not the data-generating distribution**. Because CFMNN covers spherical boundaries using rectangular boxes, the corners of the hyperboxes induce geometric interaction artifacts.

---

## 5. Global Feature Importance Aggregation

In addition to explaining individual predictions, PACE-CFMNN supports three global feature ranking methods (`pace_cfmnn/global_importance.py`):

1. **Data-Based Mean Absolute Attribution (`data_based`):**
   $$I_i^{\text{data}} = \frac{1}{N_{\text{eval}}} \sum_{n=1}^{N_{\text{eval}}} |\phi_i(x_n)|$$
   Computes global importance by averaging instance-level Shapley or flip attributions across an evaluation dataset.
2. **Normalized Mean Aggregation (`normalised_mean`):**
   $$I_i^{\text{norm}} = \frac{1}{N_{\text{eval}}} \sum_{n=1}^{N_{\text{eval}}} \frac{|\phi_i(x_n)|}{\sum_{j=1}^d |\phi_j(x_n)| + 10^{-12}}$$
   Normalizes each instance's attribution vector to sum to $1.0$ prior to averaging, preventing extreme outlier predictions from dominating global importance rankings.
3. **Model-Only Bounding Box Span (`model_only`):**
   $$I_i^{\text{model}} = \frac{1}{H} \sum_{j=1}^H \big( 1.0 - (W_{ji} - V_{ji}) \big)$$
   Evaluates global feature importance directly from hyperbox geometry **without querying a single data point**. Dimensions along which hyperboxes are tightly constrained ($W_{ji} - V_{ji} \ll 1$) indicate features that are critical for class discrimination.

### Agreement with Permutation Feature Importance (Pre-Registered B2)
In pre-registered benchmarks across 11 datasets (`results/global_importance/`):
* Averaged PACE-flip scores agreed with scikit-learn's Permutation Feature Importance significantly better than mean |KernelSHAP| on **3 of 11 datasets** (Landsat, Letter, Page Blocks), with a higher mean agreement on 10 of 11.
* PACE-flip evaluated on every single test point in milliseconds, whereas KernelSHAP was restricted to a subset of $50$ points due to computational cost limits.

---

## 6. Cross-References

- For the mathematical foundations of ramp functions and delta formulations: [`06_Mathematical_Foundations_and_Theoretical_Formulation.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/06_Mathematical_Foundations_and_Theoretical_Formulation.md)
- For the file walkthrough of `explain.py` and `counterfactual.py`: [`08_Codebase_Implementation_Deep_Dive_Core_Modules.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/08_Codebase_Implementation_Deep_Dive_Core_Modules.md)
- For testing latency and FLOP profiles: [`16_Component_Wise_Testing_Time_Latency_and_Compute_Profile.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/16_Component_Wise_Testing_Time_Latency_and_Compute_Profile.md)
- For the formal proofs and verification of Propositions 5–10: [`21_Tough_Review_Audit_and_Formal_Proof_Extensions.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/21_Tough_Review_Audit_and_Formal_Proof_Extensions.md)
- For the deep mechanics of Self-Repair and certificates: [`22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md)
