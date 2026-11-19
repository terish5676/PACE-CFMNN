# 18. Error Analysis, Failure Cases, and Limitations

## 1. Executive Summary of Limitations

A hallmarks of scientific integrity is the rigorous, transparent disclosure of failure modes, boundary conditions, and architectural trade-offs. The PACE-CFMNN framework does not claim universal superiority over unconstrained black-box models. This document examines seven fundamental failure modes and limitations identified during extensive empirical audits across the workspace (`results/FINDINGS.md` §§6–7, `docs/peer_review_paper_text.md`, `research_audit/LIMITATIONS_PLAN.md`).

```
+----------------------------------------------------------------------------------------------------+
|                                    TAXONOMY OF SYSTEM LIMITATIONS                                  |
|                                                                                                    |
|    1. The Accuracy Gap: CFMNN trails Random Forests by 0.2% - 21.5% across all 13 benchmarks.     |
|    2. Tie-Dominated Error Modality: Ties decide up to 76% of points on complex datasets.           |
|    3. Margin Explainer Blindspot at Ties: Margin is zero at ties; PACE-SHAP explains tie-entry.   |
|    4. Training Memorization on Low-D Sets: S1-S3, Wine, and Digits have 0% training errors.        |
|    5. Loose Invariance Certificates: Certified impact set is 5x - 15x larger than actual flips.    |
|    6. Condition S Practical Failure: First shrinks always cut boundary members (Gate mandatory!).  |
|    7. Exact SHAP Curse of Dimensionality: #P-hardness restricts exact mode to d <= 12 features.    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Failure Mode 1: The Predictive Accuracy Penalty

Across all 13 evaluated benchmark datasets, CFMNN is less accurate than an unconstrained Random Forest ($200$ trees) and a 10-Nearest Neighbors classifier (`results/review_v1/accuracy/`):
* **Minor Gap ($< 3.0\%$):** Shuttle ($-0.18\%$), Thyroid ($-1.24\%$), Wine ($-1.67\%$), Breast Cancer ($-2.98\%$), Digits ($-3.06\%$).
* **Moderate Gap ($5.0\% - 7.5\%$):** Skin ($-5.21\%$), Landsat ($-7.40\%$), Page Blocks ($-7.50\%$).
* **Severe Gap ($> 9.0\%$):** S1 ($-9.50\%$), S3 ($-9.50\%$), S2 ($-11.54\%$), MAGIC ($-12.71\%$), Letter ($-21.53\%$).

### Why the Gap Exists
1. **Axis-Aligned Hyperbox Rigidity:** Hyperboxes are constrained to axis-aligned rectangles in $[0, 1]^d$. Complex non-linear decision boundaries (such as diagonal hyperplanes or curved manifolds) require an excessive number of tiny hyperboxes to approximate, leading to geometric staircasing.
2. **Single-Pass Greedy Ordering:** Unlike Random Forests that aggregate hundreds of bootstrapped decorrelated trees, CFMNN trains greedily in a single pass. Early errors in box positioning cannot be corrected during training without post-hoc Self-Repair.

---

## 3. Failure Mode 2: Cross-Class Tie Dominance on Complex Datasets

A critical empirical discovery documented in `peer_review_paper_text.md` (§C) is that on complex, high-overlap datasets, the nearest-centroid tie-breaker decides a massive proportion of predictions:

### Share of Test Rows Decided by Nearest Centroid vs. Share of Errors on Ties

| Dataset | Share of All Test Rows Decided by Centroid Ties | Share of Model Test Errors Occurring on Tied Rows |
|---|---|---|
| **S1, S2, S3** | $0.0\%$ | $0.0\%$ |
| **Wine** | $0.0\%$ | $0.0\%$ |
| **Breast Cancer** | $0.8\%$ | $12.5\%$ |
| **Digits** | $0.5\%$ | $8.5\%$ |
| **Thyroid** | $5.2\%$ | **$53.0\% - 60.0\%$** |
| **Shuttle** | $12.6\%$ | **$83.3\%$** |
| **Skin** | $20.2\%$ | **$90.0\% - 100.0\%$** |
| **Landsat** | $24.8\%$ | **$58.5\%$** |
| **Letter** | **$57.2\%$** | **$95.0\% - 100.0\%$** |
| **Page Blocks** | **$72.8\%$** | **$92.0\% - 100.0\%$** |
| **MAGIC** | **$76.1\%$** | **$98.0\% - 100.0\%$** |

#### Crucial Insights on Tie Pathology
* On MAGIC, Page Blocks, and Letter, **over $90\%$ of all test errors occur on tied rows**.
* On these datasets, error is not driven by points falling outside hyperboxes where membership ramps fail; rather, it is driven by **hyperbox overlap between competing classes**, which the nearest-centroid heuristic resolves in favor of the wrong class.

---

## 4. Failure Mode 3: The Explainer Blindspot on Tied Points

Because PACE-SHAP explains the predicted class's margin:
$$g_A(x) = s'_A(x) - \max_{c \ne A} s'_c(x)$$
whenever a query point lies in an overlapping tie ($s'_A(x) = s'_B(x)$), the prediction margin is identically zero:
$$g_A(x) \equiv 0.0$$
Consequently, on a tied row, PACE-SHAP values add up to:
$$\sum_{i=1}^d \phi_i = 0.0 - \mathbb{E}_z[g_A(z)] = -\frac{1}{R} \sum_{r=1}^R g_A(z_r)$$
* **The Semantic Limitation:** On tied points, PACE-SHAP attributions explain **how features brought the two classes to an exact tie**, rather than **why the nearest-centroid distance broke the tie in favor of class $A$**.
* **The Solution (Proposition 10):** To address this limitation, we developed `PaceDecision` (`pace_cfmnn/decision.py`), which explains the hard indicator decision $\mathbb{I}(\hat{y} = A)$ directly via centroid distance logic on tied subsets.

---

## 5. Failure Mode 4: Training Set Memorization on Low-Dimensional Suites

At the default parameter setting $\theta = 0.30$:
* CFMNN achieves **$0.0\%$ training error** on synthetic S1, S2, S3, Wine, and Digits (`peer_review_paper_text.md` §B).
* However, on these datasets, the average number of training rows absorbed per hyperbox is very small:
  - **S1–S3:** $2.6$ training rows per box.
  - **Wine:** $4.5$ training rows per box.
  - **Digits:** $4.9$ training rows per box.
* **The Pathology:** With only $2.6$ to $4.9$ rows per box, the model is near the memorization threshold. Its test errors are not caused by cross-class overlaps inside boxes; they occur because test points land in empty space where an irrelevant box's ramp happens to dominate.
* **Impact on Self-Repair:** Because Self-Repair operates strictly on training errors, **it has zero training errors to repair on S1–S3, Wine, and Digits**, leaving test error unaffected on these datasets.

---

## 6. Failure Mode 5: Loose Mathematical Certificates (RQ4)

While Proposition 4 provides a $100\%$ mathematically sound certificate (guaranteeing zero prediction flips outside the impact set $C_{\text{cert}}$), the certificate is empirically loose (`results/refinement_v2_confirmatory/`):
* **Certificate Size:** On test sets, the tight certificate $C_{\text{cert}}$ flags between **$1.2\%$ and $35.0\%$ of all test instances** as potentially impacted by a single box edit.
* **Actual Prediction Flips:** In reality, only **$0.2\%$ to $3.7\%$ of test instances** actually change their predicted class.
* **The Slack:** Between $85\%$ and $95\%$ of instances inside the certificate retain identical predictions because the changed box, while entering the top group $T(x)$, does not have its centroid closer than the competing top-group boxes.
* **Explanation Certificate Looseness:** Proposition 7's certificate for PACE-SHAP explanations flags between **$99.1\%$ and $100.0\%$ of rows** because even tiny changes to a box ceiling can theoretically alter coalitional margin sums across the background distribution.

---

## 7. Failure Mode 6: Practical Failure of Condition S on First Shrinks

In Tough Review B6, Proposition 6 established "Condition S"—a theoretical invariant under which a box shrink is guaranteed to never break a correct training prediction.
However, empirical validation in `tests/test_tough_audit.py` and Run P11 revealed a fundamental geometric reality:
1. Every fitted hyperbox in CFMNN is the **tight bounding box of its absorbed training members**.
2. Therefore, for every boundary face $v_{ki}$ or $w_{ki}$, there exists at least one training member that sits directly on that boundary face.
3. Moving the boundary face inward past a misclassified point $x_r$ **always cuts at least one training member**.
4. Consequently, **Condition S fails for $98.1\%$ of candidate first shrinks** (holding for only $1.9\%$ of accepted edits).
5. **Architectural Implication:** Gated validation on acceptance rows remains strictly necessary; purely theoretical training-safe guarantees cannot replace empirical statistical gating in hyperbox editing.

---

## 8. Failure Mode 7: Scalability Limits of Exact PACE-SHAP

Proposition 5 proves that computing exact PACE-SHAP values is **#P-hard** via polynomial reduction from weighted voting games.
* Evaluating all $2^d$ feature coalitions scales exponentially.
* At $d = 12$, exact enumeration evaluates $4,096$ coalitions in $45$ ms.
* At $d = 16$ (Letter), exact enumeration requires $65,536$ coalitions, taking **$26.1$ seconds per point** even with the dynamic recurrence engine.
* Beyond $d = 12$, exact mode must be abandoned in favor of antithetic permutation sampling.

---

## 9. Master Claims Ledger: What We Can and Cannot Claim

To ensure absolute adherence to peer-review standards, the table below delineates verified claims from impermissible exaggerations:

```
+----------------------------------------------------------------------------------------------------+
|                               MASTER SCIENTIFIC POSITIONING LEDGER                                 |
+----------------------------------------------------+-----------------------------------------------+
| WHAT WE CAN RIGHTFULLY CLAIM                       | WHAT WE MUST NEVER CLAIM                      |
+----------------------------------------------------+-----------------------------------------------+
| 1. Exact, closed-form Shapley values for d <= 12.  | 1. That CFMNN outperforms Random Forests or   |
| 2. 1.000000000000 deterministic stability.         |    EBMs in raw classification accuracy.       |
| 3. Never less faithful than KernelSHAP on margin.  | 2. That exact PACE-SHAP runs in polynomial    |
| 4. PACE-flip is 25x-134x faster than occlusion.    |    time for arbitrary dimensions (#P-hard!).  |
| 5. Every training error is a cross-class tie.      | 3. That Self-Repair eliminates the need for   |
| 6. Self-repair adds +1.5 to +6.0 macro-F1 with     |    labeled validation data (Gate needs them!).|
|    ZERO new labels.                                | 4. That certificates tightly predict exact    |
| 7. Zero prediction changes outside certificates    |    flip locations (they are loose bounds).    |
|    across 46,153 empirical edits.                  | 5. That PACE-SHAP explains centroid ties      |
|                                                    |    (it explains the zero margin at ties).     |
+----------------------------------------------------+-----------------------------------------------+
```

---

## 10. Cross-References

- For the base model architecture and expansion rules: [`07_Component_By_Component_Architectural_Design.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/07_Component_By_Component_Architectural_Design.md)
- For the full empirical benchmark results across all models: [`13_Overall_Benchmark_Results_and_SOTA_Comparison.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/13_Overall_Benchmark_Results_and_SOTA_Comparison.md)
- For the formal academic audit and paper blueprint: [`19_Research_Readiness_Audit_and_Paper_Blueprint.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/19_Research_Readiness_Audit_and_Paper_Blueprint.md)
- For the formal mathematical proofs of #P-hardness and Condition S: [`21_Tough_Review_Audit_and_Formal_Proof_Extensions.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/21_Tough_Review_Audit_and_Formal_Proof_Extensions.md)
- For the deep mechanics of Self-Repair and certificates: [`22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md)
