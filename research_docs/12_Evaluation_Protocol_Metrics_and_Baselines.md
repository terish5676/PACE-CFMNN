# 12. Evaluation Protocol, Metrics, and Baselines

## 1. Experimental Rigor and Protocol Standardization

To prevent data snooping, p-hacking, and post-hoc rationalization, all primary experimental suites in PACE-CFMNN followed strict pre-registration protocols. Hypotheses, metric definitions, seed ranges, and stopping rules were written into `PREREGISTRATION.md` files and cryptographically hashed (`release/osf/sha256sums.txt`) prior to launching compute jobs.

### Experimental Seed Allocations & Partitioning Protocols

| Experiment Suite | Seed Range | Protocol Version | Pre-Registration Record | Core Focus |
|---|---|---|---|---|
| **Base Model G0** | Seeds 0–4 | Exploratory G0 | `results/g0/summary.md` | Single-model parity vs. PReMI 2023 ensemble |
| **RQ1 Faithfulness (v2)**| Seeds 0–9 | Fair Benchmark v2 | `results/faithfulness/` | Signed margin attributions: PACE-flip vs. SHAP |
| **RQ1 PACE-SHAP (v3)** | Seeds 30–39 | Pre-registered v3 | `faithfulness_v3/PREREGISTRATION.md` | Primary test: PACE-SHAP vs. KernelSHAP |
| **RQ2 Trust Signals** | Seeds 10–19 | Confirmatory RQ2 | `error_detection_confirmatory/` | Intrinsic signals (PD, EI, $U^*$) vs. baselines |
| **RQ2 v2 Learned Combiner**| Seeds 20–29 | Confirmatory RQ2 v2 | `error_detection_v2/` | Multimodal logistic regression error detection |
| **RQ1 Extensions (B1–B4)**| Seeds 40–49 | Pre-registered B1–B4 | `extensions_v1/PREREGISTRATION.md` | Sparse CFs, global aggregation, conditional SHAP |
| **RQ3/RQ4 Self-Repair** | Seeds 50–59 | Pre-registered Conf. 2| `refinement_v2_confirmatory/` | Zero-label repair vs. active learning & retraining |
| **Readable Baselines** | Seeds 70–89 | Pre-registered RB | `readable_baselines/PREREGISTRATION.md`| CFMNN vs. EBM, Trees, LogReg, GFMM (20 seeds) |
| **Repair at Tuned $\theta$**| Seeds 70–89 | Pre-registered RT | `repair_theta/PREREGISTRATION.md` | Interaction of validation-chosen $\theta$ and repair |
| **Tough Review Runs** | Seeds 250–399 | Batches 1, 2, 3 | `research_audit/TOUGH_REVIEW_PLAN.md` | Hardness, GFMM, Covertype, OpenML held-out |

---

## 2. Evaluation Metrics: Mathematical Formulations & Intuitions

### 2.1 Explainability Faithfulness: Deletion & Insertion AUC

Rather than relying on human visual inspection, feature attributions are evaluated quantitatively using sequential masking on the model's prediction margin $g_A(x) = s'_A(x) - \max_{c \ne A} s'_c(x)$.
Features are sorted by signed attribution $\phi_i$ in descending order (most supportive of predicted class $A$ first):
$$\pi = (\pi_1, \pi_2, \dots, \pi_d), \quad \phi_{\pi_1} \ge \phi_{\pi_2} \ge \dots \ge \phi_{\pi_d}$$

```
+----------------------------------------------------------------------------------------------------+
|                               FAITHFULNESS CURVE EVALUATION DYNAMICS                               |
|                                                                                                    |
|    Margin g_A(x)                                                                                   |
|       ^                                                                                            |
|   1.0 |===*                                                                                        |
|       |    \                                                                                       |
|   0.5 |     *---*                                                                                  |
|       |          \                                                                                 |
|   0.0 +-----------\-----*--------------------> Step k (Features Replaced)                         |
|       |            \   /                                                                           |
|  -0.5 |             *-*     Area Under Deletion Curve (AUC)                                        |
|       |                     Lower AUC = Margin drops faster = MORE FAITHFUL!                       |
|  -1.0 +-------------------------------------------------------------------->                       |
|           k=0       k=1      k=2      k=3      ...      k=d                                        |
+----------------------------------------------------------------------------------------------------+
```

#### 1. Deletion Curve & Deletion AUC (Primary Metric)
* **Mechanics:** Starting with original point $x$, step $k \in \{0, \dots, d\}$ replaces the top $k$ features with coordinates from a random training background row $z$:
  $$x^{(k)} = \big( z_{\{\pi_1, \dots, \pi_k\}}, \, x_{-\{\pi_1, \dots, \pi_k\}} \big)$$
* **Mathematical Definition:**
  $$\text{AUC}_{\text{del}} = \frac{1}{d} \sum_{k=0}^d g_A\big(x^{(k)}\big) - \frac{1}{2d} \Big( g_A(x^{(0)}) + g_A(x^{(d)}) \Big)$$
* **Intuition:** If an explainer correctly identified the features driving class $A$, removing them must cause the predicted margin to collapse rapidly. **Lower Deletion AUC signifies superior faithfulness.**

#### 2. Insertion Curve & Insertion AUC
* **Mechanics:** Starting from an uninformative background row $z$, step $k$ reintroduces the top $k$ features from query point $x$:
  $$z^{(k)} = \big( x_{\{\pi_1, \dots, \pi_k\}}, \, z_{-\{\pi_1, \dots, \pi_k\}} \big)$$
* **Intuition:** Introducing the most critical features must rebuild the winning margin rapidly. **Higher Insertion AUC signifies superior faithfulness.**

---

### 2.2 Explainer Stability
Evaluates attribution repeatability across identical inputs:
$$\text{Stability} = \text{SpearmanRankCorr}\big( \phi^{(1)}(x), \, \phi^{(2)}(x) \big)$$
* **Closed-Form Guarantee:** For exact PACE-SHAP ($d \le 12$) and PACE-flip, Stability $\equiv \mathbf{1.000000000000}$.
* **KernelSHAP Pathology:** Repeated sampling runs yield stability scores between $0.15$ and $0.73$, exposing high estimation variance.

---

### 2.3 Error Detection Metrics: AUPRC and AUROC

When ranking test instances by predicted risk of misclassification using trust signals (PD, EI, $U^*$, PRS):
* **Ground-Truth Binary Target:** $e_i = \mathbb{I}(\hat{y}(x_i) \ne y_i) \in \{0, 1\}$.
* **Area Under the Precision-Recall Curve (AUPRC):**
  $$\text{AUPRC} = \sum_{k=1}^N \text{Precision}(k) \cdot \Delta \text{Recall}(k)$$
  *Why AUPRC Matters:* Classification errors are highly imbalanced (e.g., $0.2\%$ on Shuttle, $1.6\%$ on Thyroid). In extreme class imbalance, AUROC yields deceptively optimistic scores, whereas AUPRC rigorously measures true error-capture utility.

---

### 2.4 Counterfactual Recourse Metrics

Given query point $x$ and counterfactual candidate $x_{\text{cf}}$ satisfying $\hat{y}(x_{\text{cf}}) = \text{target}$:
1. **L1 Distance (Cost of Recourse):**
   $$\text{Cost}(x, x_{\text{cf}}) = \|x - x_{\text{cf}}\|_1 = \sum_{i=1}^d |x_i - x_{\text{cf}, i}|$$
2. **Sparsity ($L_0$ Feature Count):**
   $$\text{Sparsity} = \sum_{i=1}^d \mathbb{I}\big( |x_i - x_{\text{cf}, i}| > 10^{-9} \big)$$
   Lower sparsity means fewer lifestyle or operational changes required by the user.
3. **Plausibility (Manifold Proximity):**
   $$\text{Plausibility}(x_{\text{cf}}) = \min_{r : y_r = \text{target}} \|x_{\text{cf}} - x_r^{\text{train}}\|_1$$
   Measures distance to the nearest real observed training point of the target class. Smaller distance ensures the counterfactual is realistic rather than an adversarial artifact.

---

### 2.5 Repair & Invariance Metrics (RQ3 / RQ4)

1. **Fix Share:**
   $$\text{Fix Share} = \frac{N_{\text{fixed}}}{N_{\text{fixed}} + N_{\text{broken}}} \in [0, 1]$$
   where $N_{\text{fixed}}$ is the count of test points transitioning from wrong to right ($\hat{y}_{\text{before}} \ne y \land \hat{y}_{\text{after}} = y$), and $N_{\text{broken}}$ is the count of regressions ($\hat{y}_{\text{before}} = y \land \hat{y}_{\text{after}} \ne y$).
2. **Broken Share:**
   $$\text{Broken Share} = \frac{N_{\text{broken}}}{N_{\text{test}}}$$
   Measures absolute predictive churn across deployed instances.
3. **Non-Flip Rate (NFR):** Fraction of predictions that remain completely unchanged.
4. **Certificate Leakage Rate:**
   $$\text{Leakage} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\Big( \big(\hat{y}(x_i; M') \ne \hat{y}(x_i; M)\big) \land (x_i \notin C_{\text{cert}}) \Big) \equiv \mathbf{0.0000\%}$$

---

## 3. Detailed Baseline Inventory

### 3.1 Explainability Baselines
1. **KernelSHAP (Lundberg & Lee 2017):**
   - *Usual Setting:* $200$ coalitional subset samples, $20$ background rows, $\ell_1$-regularized feature selection keeping $10$ features (standard `shap` package default).
   - *Strong Setting:* $2,048$ samples, no feature selection (exact enumeration for $d \le 11$).
2. **LIME (Ribeiro et al. 2016):**
   - Tabular perturbation explainer fitting an interpretable ridge regression surrogate over $1,000$ perturbed samples. Configured with target set to prediction margin $g_A(x)$ for fair benchmarking.
3. **Brute-Force Feature Occlusion:**
   - Evaluates decision impact by iterating through $32$ quantile replacements for each feature in individual forward passes.

---

### 3.2 Counterfactual Baselines
1. **DiCE (Mothilal et al. 2020):**
   - Diverse Counterfactual Explanations via optimization. Configured fairly with 5 candidate generations, sparse post-processing, and nearest valid selection (`results/counterfactuals/`). Evaluated in both Random Search and Genetic Search modes.

---

### 3.3 Interpretable & High-Capacity Classifier Baselines

To contextualize CFMNN accuracy, the codebase evaluates six external model families across identical splits:
1. **Explainable Boosting Machine (EBM / `interpret-core`):** A modern Generalized Additive Model with pairwise interactions ($GA^2M$) trained via cyclical gradient boosting: $g(E[y]) = \beta_0 + \sum f_i(x_i)$.
2. **Decision Tree (`DecisionTreeClassifier`):** Scikit-learn CART decision tree with hyperparameter depth tuned by 3-fold cross-validation.
3. **Logistic Regression (`LogisticRegression`):** Standard linear baseline with $\ell_2$ penalty.
4. **General Fuzzy Min-Max Network (GFMM / `hyperbox-brain`):** Incremental hyperbox network using Simpson's min-membership operator and heuristic contraction.
5. **Random Forest (`RandomForestClassifier`):** SOTA non-linear ensemble baseline with $200$ trees.
6. **10-Nearest Neighbors (`KNeighborsClassifier`):** Non-parametric prototype baseline with $k = 10$.

---

## 4. Fairness Controls & Statistical Significance Testing

To ensure rigorous academic standards, all comparisons adhere to the following statistical protocols:

1. **Paired Two-Sided Wilcoxon Signed-Rank Test:**
   Evaluates paired differences over the 10 experimental seeds (`scipy.stats.wilcoxon`).
2. **Holm-Bonferroni Family-Wise Error Correction:**
   Across a run's $K$ benchmark datasets, $p$-values are sorted: $p_{(1)} \le p_{(2)} \le \dots \le p_{(K)}$. A comparison is declared statistically significant if and only if:
   $$p_{(i)} < \frac{0.05}{K - i + 1}$$
3. **Nadeau-Bengio Resampled $t$-Test Correction (Tough Review D1):**
   Standard $t$-tests assume independent splits, underestimating variance when splits overlap. The Nadeau-Bengio correction adjusts the sample variance by factor:
   $$\sigma^2_{\text{corrected}} = \left( \frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}} \right) \cdot S^2$$
   For an 80/20 split ($n_{\text{test}}/n_{\text{train}} = 0.25$) over $J = 10$ seeds, this multiplies the variance by $3.5\times$, eliminating spurious claims of significance (`results/tough_review/reanalysis.md`).

---

## 5. Cross-References

- For the full empirical benchmark results across all metrics: [`13_Overall_Benchmark_Results_and_SOTA_Comparison.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/13_Overall_Benchmark_Results_and_SOTA_Comparison.md)
- For the granular component-wise ablation comparisons: [`14_Component_Wise_Accuracy_and_Ablation_Study.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/14_Component_Wise_Accuracy_and_Ablation_Study.md)
- For statistical significance tables and Nadeau-Bengio corrected tests: [`17_Statistical_Significance_Robustness_and_Stability.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/17_Statistical_Significance_Robustness_and_Stability.md)
- For peer-review audit and checklist compliance: [`19_Research_Readiness_Audit_and_Paper_Blueprint.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/19_Research_Readiness_Audit_and_Paper_Blueprint.md)
