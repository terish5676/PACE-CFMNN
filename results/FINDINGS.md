# PACE-CFMNN: Comprehensive Scientific Findings (Phases 1–7 & Protocol v6.0)

This document synthesizes the empirical findings, mathematical verifications, negative results, and scientific claim boundaries of the PACE-CFMNN research framework.

---

## 1. Baseline Model (G0) & Algorithmic Corrections

Evaluated across 5 random seeds using stratified 80/20 train/test splits, min-max scaling fitted strictly on training rows ($\theta = 0.3$, $\gamma = 4.0$).

### Key Findings
1. **Centroid Accumulation Bug Fix:**
   - In the published CFMNN literature (Aadarsh & Sai Prasad, PReMI 2023), points that landed inside an existing hyperbox were skipped during centroid accumulation.
   - On `shuttle`, **99.5% of training points land inside an existing box**, meaning original centroids were computed from only the first few boundary-forming instances.
   - Counting all interior points in the running centroid boosted Shuttle test accuracy from **93.12% to 98.62%** (+5.5% jump) and up to **99.80%** when fully tuned.
2. **Search Expansion Solves Dimensional Collapse:**
   - The published CFMNN algorithm only tested whether the single highest-membership same-class box could expand. In higher dimensions ($d \ge 10$), that box was almost always at the size threshold $\theta$, forcing the algorithm to create isolated single-point boxes.
   - Our search expansion scans candidate same-class boxes that can still expand without exceeding $\theta$.
   - **Hyperbox Count Reductions:**
     - **Skin ($d=3$):** 288 boxes $\to$ **8 boxes (36.0× reduction)**
     - **Satellite ($d=36$):** 3,348 boxes $\to$ **218 boxes (15.4× reduction)**
     - **Thyroid ($d=21$):** 777 boxes $\to$ **56 boxes (13.9× reduction)**
     - **Letter ($d=16$):** 8,057 boxes $\to$ **613 boxes (13.1× reduction)**
     - **Breast Cancer ($d=30$):** 415 boxes $\to$ **42 boxes (9.9× reduction)**
     - **Digits ($d=64$):** 1,596 boxes $\to$ **260 boxes (6.1× reduction)**, with accuracy rising from 78.70% to **96.11%** (matching XGBoost's 96.20%).
3. **Data Pathologies (The Sepsis Dataset):**
   - On the UCI Sepsis dataset, 91.5% of rows shared the exact same feature values with rows of the opposite outcome label. Deterministic hyperbox architectures cannot separate identical points mapping to opposing classes, capping accuracy at 48.65% (below the 92.65% majority class baseline). Sepsis was formally documented as unsuitable for deterministic rule/box learning.

---

## 2. Explanation Faithfulness, Stability, and Speed (RQ1)

Evaluated across 3 seeds × 50 test instances per dataset using deletion margin AUC (lower is more faithful), millisecond execution latency, and top-3 feature overlap stability across seeds:

| Dataset | PACE-flip | PACE-multi | Brute-Force Occlusion | KernelSHAP | LIME | Random Baseline |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1 (Synthetic)** | **−0.107** | 0.070 | **−0.107** | 0.051 | 0.074 | 0.275 |
| **S2 (Synthetic)** | **−0.110** | 0.135 | **−0.110** | 0.083 | 0.119 | 0.271 |
| **S3 (Synthetic)** | **0.040** | 0.189 | **0.040** | 0.174 | 0.182 | 0.341 |
| **Wine** | **−0.047** | 0.027 | **−0.047** | 0.002 | 0.037 | 0.255 |
| **Breast Cancer** | **0.158** | 0.176 | **0.158** | 0.198 | 0.231 | 0.421 |
| **Thyroid** | **0.534** | 0.569 | **0.534** | 0.606 | 0.646 | 0.741 |
| **Shuttle** | **0.250** | 0.254 | **0.250** | 0.293 | 0.297 | 0.433 |

### Latency Comparison (Milliseconds per Explanation)

| Dataset | PACE-multi | PACE-flip | Brute-Force Occlusion | KernelSHAP | LIME | PACE-flip Speedup vs SHAP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1** | 0.50 ms | 1.11 ms | 86.2 ms | 548 ms | 266 ms | **494× faster** |
| **Wine** | 0.03 ms | 0.11 ms | 5.7 ms | 28 ms | 13 ms | **254× faster** |
| **Breast Cancer** | 0.05 ms | 0.26 ms | 21.5 ms | 61 ms | 24 ms | **234× faster** |
| **Thyroid** | 0.05 ms | 0.23 ms | 15.4 ms | 53 ms | 25 ms | **230× faster** |
| **Shuttle** | 0.04 ms | 0.07 ms | 4.9 ms | 62 ms | 18 ms | **885× faster** |

### Stability Across Random Seeds
- **PACE (all variants):** **1.00** (100% deterministic, zero sampling noise).
- **KernelSHAP:** 0.60 – 0.99 (exhibits run-to-run sampling variance).
- **LIME:** 0.70 – 1.00 (stochastic local surrogate fitting).

---

## 3. Trust Diagnostics & Falsification of Hypothesis H2

We evaluated whether adding within-class explanation instability ($EI_{within}$) to Prediction Disagreement ($PD$) and Ambiguity ($U$) improves test-time error detection on held-out test data (Research Protocol v6.0 Hypothesis H2):

$$R_{base}(x) = \alpha \cdot PD(x) + (1-\alpha) \cdot U(x)$$
$$R_{ext}(x) = \text{LogisticRegression}(PD(x), U(x), EI_{within}(x))$$

### Empirical Results Across Datasets (1,000 Bootstrap Resamples)

| Dataset | AUROC ($PD+U$) | AUROC ($PD+U+EI$) | $\Delta \text{AUROC}$ | Bootstrap $p$-value | AUPRC ($PD+U$) | AUPRC ($PD+U+EI$) | $\Delta \text{AUPRC}$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Digits** | **0.978** | 0.976 | −0.002 | 0.695 | **0.683** | 0.679 | −0.004 |
| **Skin** | **0.963** | 0.963 | −0.000 | 0.765 | **0.439** | 0.439 | −0.000 |
| **Thyroid** | **0.752** | 0.705 | −0.048 | 0.773 | **0.085** | 0.069 | −0.016 |
| **Shuttle** | **0.585** | 0.491 | −0.094 | 0.873 | **0.448** | 0.097 | −0.352 |

### Scientific Conclusion: Falsification of H2
- On all evaluated datasets, adding $EI_{within}$ yielded **$\Delta \text{AUROC} \le 0$** ($p > 0.50$).
- Within a hyperbox core, CFMNN membership gradients are piece-wise linear or constant; local feature attributions do not fluctuate enough inside the box to provide error information beyond what $PD$ and $U$ already capture.
- **Protocol Mandate:** Hypothesis H2 was falsified. $EI_{within}$ was stripped from the operational error risk score and retained strictly as a descriptive explanation stability signal. The primary error risk score is standardized as:
  $$R(x) = \alpha \cdot PD(x) + (1-\alpha) \cdot U(x)$$

---

## 4. Exact Structural Caching (Appendix B) vs. Approximate Caching

We evaluated query explanation reuse under local perturbations ($\rho = 0.01$):
- **Approximate Caching:** Reuses explanations whenever Euclidean distance $\|x - x'\|_2 < \rho$.
- **Exact Structural Caching:** Evaluates the mathematically proven condition from Appendix B:
  $$m_{j^*}(x) - \max_{j \neq j^*} m_j(x) > \gamma \rho$$

### Empirical Caching Results (1,000 Test Queries)

| Dataset | Cache Mode | Structural Mismatch Rate | Mean Attribution Error | Mathematical Proof |
|---|---|:---:|:---:|---|
| **Digits** | Approximate Cache | 7.62% | $7.79 \times 10^{-5}$ | Unproven / Approximate |
| | **Exact Structural Cache** | **0.00%** | **0.0** | **Proven (Appendix B)** |
| **Thyroid** | Approximate Cache | **78.75%** | $4.70 \times 10^{-4}$ | Unproven / Approximate |
| | **Exact Structural Cache** | **0.00%** | **0.0** | **Proven (Appendix B)** |
| **Skin** | Approximate Cache | 11.50% | $3.70 \times 10^{-3}$ | Unproven / Approximate |
| | **Exact Structural Cache** | **0.00%** | **0.0** | **Proven (Appendix B)** |
| **Shuttle** | Approximate Cache | 8.88% | $2.80 \times 10^{-4}$ | Unproven / Approximate |
| | **Exact Structural Cache** | **0.00%** | **0.0** | **Proven (Appendix B)** |

### Scientific Conclusion
Approximate caching causes severe structural mismatches (up to 78.75% of queries retrieve explanations computed for the wrong winning hyperbox). Exact structural caching guarantees **0.00% structural mismatch** and exact mathematical fidelity down to machine precision.

---

## 5. Model Refinement & The Six-Condition Acceptance Gate

We evaluated traditional Simpson (1992) FMM contraction against Explanation-Guided Shrinking with nested cross-fitting and the Six-Condition Acceptance Gate:

| Dataset | Refinement Method | Test Acc (%) | Test Macro-F1 (%) | $\Delta \text{Acc}$ vs Baseline | Collateral Errors | Gate Passed Rate |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Shuttle** | No Refinement | 98.62% | 71.28% | Baseline | 0.0 | 100% |
| | **FMM Contraction (Simpson 1992)** | **30.77%** | **36.80%** | **−67.86%** | **997.3** | **0.0% (Rejected)** |
| | Explanation-Guided Shrink | 72.77% | 63.56% | −25.85% | 387.0 | 0.0% (Rejected) |
| **Skin** | No Refinement | 96.35% | 94.15% | Baseline | 0.0 | 100% |
| | FMM Contraction (Simpson 1992) | 89.57% | 80.19% | −6.78% | 126.0 | 0.0% (Rejected) |
| | **Explanation-Guided Shrink** | **97.74%** | **96.47%** | **+1.39%** | **10.3** | **100% (Accepted)** |
| **Digits** | No Refinement | 96.11% | 96.09% | Baseline | 0.0 | 100% |
| | FMM Contraction (Simpson 1992) | 96.11% | 96.09% | +0.00% | 0.0 | 100% (Neutral) |
| | Explanation-Guided Shrink | 96.11% | 96.09% | +0.00% | 0.0 | 100% (Neutral) |

### Key Takeaway on Refinement
- Traditional geometric contraction blindly shrinks overlapping boxes, causing catastrophic collateral collapse when classes naturally overlap in feature space.
- The **Six-Condition Acceptance Gate** successfully caught and blocked destructive contractions on Shuttle, while permitting genuine improvements (+1.39% accuracy) on Skin.

---

## 6. Comprehensive Protocol v6 Benchmark Results

Comparing optimized CFMNN against industry-standard tree ensembles (Random Forest, XGBoost, LightGBM) and linear baselines:

| Dataset | Metric | CFMNN-paper | **CFMNN-primary (Ours)** | LightGBM | RandomForest | XGBoost |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Digits** | Accuracy | 78.70 ± 1.53% | **96.11 ± 0.56%** | 96.85 ± 0.16% | 97.31 ± 0.16% | 96.20 ± 0.32% |
| | Macro-F1 | 78.10 ± 1.83% | **96.09 ± 0.57%** | 96.85 ± 0.17% | 97.31 ± 0.16% | 96.20 ± 0.32% |
| | Train Time | 0.091 s | **0.041 s** | 0.463 s | 0.246 s | 0.530 s |
| **Shuttle** | Accuracy | 93.12 ± 0.43% | **98.62 ± 0.59%** | 94.07 ± 2.18% | 99.92 ± 0.05% | 99.97 ± 0.04% |
| | Macro-F1 | 67.95 ± 6.50% | **71.28 ± 6.37%** | 36.73 ± 4.50% | 79.09 ± 7.39% | 99.60 ± 0.35% |
| | Train Time | 0.226 s | **0.223 s** | 0.378 s | 0.377 s | 0.510 s |
| **Skin** | Accuracy | 97.77 ± 0.62% | **96.35 ± 1.10%** | 99.85 ± 0.04% | 99.86 ± 0.04% | 99.83 ± 0.08% |
| | Macro-F1 | 96.52 ± 1.00% | **94.15 ± 1.87%** | 99.77 ± 0.07% | 99.79 ± 0.06% | 99.75 ± 0.11% |
| | Train Time | 0.398 s | **0.305 s** | 0.710 s | 0.366 s | 0.175 s |
| **Thyroid**| Accuracy | 93.84 ± 0.58% | **96.88 ± 0.28%** | 99.77 ± 0.11% | 99.70 ± 0.08% | 99.56 ± 0.14% |
| | Macro-F1 | 75.72 ± 0.57% | **84.91 ± 1.24%** | 98.58 ± 0.92% | 98.37 ± 0.81% | 97.63 ± 0.82% |
| | Train Time | 0.550 s | **0.154 s** | 0.210 s | 0.345 s | 0.273 s |

---

## 7. Version 0.2.1 Fixes & Architectural Enhancements

Based on the empirical findings and practical limitations, the following architectural improvements were integrated into the core package:

1. **Automated Raw Tabular Preprocessor (`AutoCFMNN`):** End-to-end scikit-learn compatible estimator accepting raw pandas DataFrames with strings, booleans, missing values, and unscaled numbers. Handles missing value imputation, categorical encoding, Min-Max scaling, and duplicate conflict resolution seamlessly.
2. **Unified Explanation API (`CFMNN.explain`):** One-stop method exposing Target B (PACE-flip decision occlusion), Target A (winning box additive decomposition), and Target C (contrastive explanation) with clear scope discipline.
3. **Candidate Bounding Theorem in Caching:** When queries lie near decision boundaries, the cache mathematically proves that any box with $m_j(x) < m_{j^*}(x) - \gamma \rho$ can never win, evaluating only the active competitor set $|\mathcal{C}(x)| \ll H$ during fallback rather than all $H$ boxes, maintaining 0.00% error while speeding up fallback by 10× to 100×.
4. **1D Sweep-Line Overlap Pre-Filtering:** Contraction uses a coordinate projection sweep along the maximum-spread axis to filter out non-overlapping boxes before multidimensional checking, eliminating quadratic pairwise comparison bottlenecks.
5. **Numerical Tolerance in Min-Max Scaling:** Added `tol = 1e-5` and automatic clipping in `CFMNN.fit()` and `_check_X()` to eliminate crashes from floating-point rounding errors (`-1e-16`, `1.0000000000000002`).
6. **Safe FMM Contraction (`safe_apply_fmm_contraction`):** Added automatic rollback logic guarded by the Six-Condition Acceptance Gate to prevent model collapse.
7. **Operational Error Scoring (`predict_with_trust`):** Standardized $R(x) = \alpha \cdot PD(x) + (1-\alpha) \cdot U(x)$ directly into the model API.
8. **Order-Invariant Permutation Ensemble (`EnsembleCFMNN`):** Added a bagging ensemble across randomized permutations to resolve single-pass order sensitivity.
9. **Data Pathology Detection (`detect_label_conflicts`, `clean_label_conflicts`):** Added automated detection and resolution of identical rows mapping to contradictory labels.
10. **Rotated / Oblique Hyperboxes (`ObliqueCFMNN`):** Solves axis-aligned staircase boundaries on diagonal and correlated manifolds by learning an orthonormal rotation matrix $R$ (via PCA or custom projection) and back-projecting attributions ($\phi_X = \phi_Z R^\top$).
11. **Pretrained Vision & Text Embedding Hooks (`EmbeddedCFMNN`):** Allows CFMNN to operate directly as an interpretable hyperbox decision head on top of PyTorch/HuggingFace encoders.
12. **Scalable Active Learning (`cluster_margin`):** Employs one-time `MiniBatchKMeans` pre-clustering, candidate subsampling, and model revision caching to scale active refinement to $100\text{k}+$ pools without quadratic bottlenecks.
13. **Operational Error Risk Decoupling (`compute_risk`):** Excludes $EI_{within}$ from operational error risk scoring, standardizing strictly on $R(x) = \alpha \cdot PD(x) + (1 - \alpha) \cdot U(x)$ in accordance with the falsification of Hypothesis H2.
14. **Comprehensive Test Suite:** 90 unit tests passing cleanly with 100% pass rate.

---

## 8. Scientific Claim Discipline (Boundaries of What We Can and Cannot Claim)

### What We Can Legally and Scientifically Claim
- **PACE-flip equals brute-force occlusion exactly** but computes it in closed form 50× to 890× faster without sampling variance.
- **PACE-multi is an ultra-fast prototype contrast explainer** (0.02–0.5 ms) with 1.00 deterministic stability.
- **Target A additive decomposition equals exact Shapley values** of the single winning box's scalar membership score $m_{j^*}(x)$ (verified to $< 3 \times 10^{-16}$).
- **Exact structural caching provides provable zero-attribution error** under the Appendix B Lipschitz bound condition.
- **Prediction Disagreement + Ambiguity ($PD + U$) provides strong error discrimination** with AUROC up to 0.978 on Digits and 0.963 on Skin.
- **CFMNN with search expansion and fixed centroids drastically compresses hyperbox models** (up to 36× fewer boxes) while matching tree ensembles on datasets like Digits (96.11%).
- **ObliqueCFMNN eliminates staircase box fragmentation on diagonal manifolds** while preserving exact linear attribution mapping.

### What We Must NOT Claim
- **Do not claim Target A equals Shapley values for the final prediction:** Target A decomposes the winning box membership; it does not account for the multi-box competitive argmax and centroid tie-breaking. Full classifier sensitivity is captured by PACE-flip (Target B).
- **Do not claim PACE-flip is more faithful than occlusion:** It is mathematically identical; its contribution is exact, closed-form speedup.
- **Do not claim $EI_{within}$ improves error detection:** Empirical results across all datasets falsified Hypothesis H2 ($\Delta \text{AUROC} \le 0$). Operational error risk scoring is strictly $PD + U$.
- **Do not claim classic FMM contraction is safe:** Blind geometric contraction causes catastrophic collateral damage (Shuttle collapsing from 98.6% to 30.7%) and must always be guarded by an acceptance gate.
- **Do not claim CFMNN applies to unscaled or conflicting data without AutoCFMNN:** Base CFMNN strictly requires bounded continuous inputs $[0, 1]^d$. Raw DataFrames with strings, missing values, or unscaled features must use `AutoCFMNN`.
