# 14. Component-Wise Accuracy and Ablation Study

## 1. Scope & Methodology of Ablation Analysis

Ablation studies isolate the causal impact of individual algorithmic choices, verifying whether performance gains stem from genuine mathematical principles or confounding implementations. In this document, we walk through the exact component-wise deltas ($\Delta$) across all system layers:
1. **Base Architecture Ablations:** Centroid rule (`"fixed"` vs `"original"`) and expansion rule (`"search"` vs `"best"`).
2. **Explainability Ablations:** Interventional vs conditional games, attribution pruning, and global aggregation modes.
3. **Trust Telemetry Ablations:** Stepwise combination from raw margin uncertainty to composite $PD + U^*$ and learned combiners.
4. **Self-Repair Ablations:** Surgical face cut choosers, gate thresholds, and comparison against Simpson's contraction.

---

## 2. Base Model Architectural Ablations (Gate G0)

In Phase 1, we evaluated the four combinations of centroid accumulation and expansion rules against the original PReMI 2023 paper across 5 seeds (`results/g0/summary.md`):

```
+----------------------------------------------------------------------------------------------------+
|                                    BASE ARCHITECTURE ABLATION TREE                                 |
|                                                                                                    |
|    Level 0: Paper as Written (Original Centroids, Best Expansion)                                  |
|         |                                                                                          |
|         +---> Level 1A: + Fixed Centroids Alone (All absorbed points update sum/count)              |
|         |                                                                                          |
|         +---> Level 1B: + Search Expansion Alone (Search all same-class boxes under theta)          |
|         |                                                                                          |
|         +---> Level 2:  + Both Architectural Fixes (Full Main CFMNN Model)                         |
+----------------------------------------------------------------------------------------------------+
```

### Granular Accuracy & Box Count Deltas (Mean Across 5 Seeds)

| Dataset | Level 0: Paper Rules (Orig/Best) | Level 1A: +Fixed Centroids Alone | Level 1B: +Search Expansion Alone | Level 2: Full Model (Fixed/Search) | Net Delta ($\Delta_{\text{Full} - \text{Paper}}$) |
|---|---|---|---|---|---|
| **Shuttle** | $93.33\%$ ($38$ boxes) | **$99.80\%$** ($38$ boxes) | $93.33\%$ ($38$ boxes) | **$99.80\%$** ($38$ boxes) | **$+6.47\%$ Accuracy** |
| **Digits** | $81.72\%$ ($585$ boxes) | $81.75\%$ ($585$ boxes) | $95.31\%$ ($232$ boxes) | **$95.33\%$** ($232$ boxes) | **$+13.61\%$ Acc, $-60\%$ Boxes** |
| **Thyroid** | $95.04\%$ ($124$ boxes) | $95.12\%$ ($124$ boxes) | $97.15\%$ ($48$ boxes) | **$97.18\%$** ($48$ boxes) | **$+2.14\%$ Acc, $-61\%$ Boxes** |
| **S1** | $78.92\%$ ($3,748$ boxes) | $78.95\%$ ($3,748$ boxes)| $88.70\%$ ($285$ boxes) | **$88.74\%$** ($285$ boxes) | **$+9.82\%$ Acc, $-92\%$ Boxes** |
| **Skin** | **$96.88\%$** ($1,885$ boxes)| $97.01\%$ ($1,885$ boxes)| $94.68\%$ ($86$ boxes) | $94.74\%$ ($86$ boxes) | $-2.14\%$ Acc, $-95\%$ Boxes |

#### Causal Analysis of Findings
1. **The Shuttle Breakthrough ($+6.47\%$):** On Shuttle, $99.5\%$ of points land inside an existing hyperbox. Under the paper's original rule, these points were skipped. When boxes tied during inference ($12.6\%$ of test points), centroids were unrepresentative, resolving ties incorrectly. Updating centroids on all absorbed points resolved these ties completely.
2. **The Search Expansion Breakthrough ($+13.61\%$ on Digits, $+9.82\%$ on S1):** In spaces with $d \ge 10$, the single highest-membership box is almost always at capacity. Best-only expansion triggered degenerate point-box proliferation ($3,748$ boxes on S1). Search expansion allowed points to grow non-maximal same-class boxes, reducing box counts by up to $13\times$ while expanding generalization boundaries.

---

## 3. Explainability Ablations (RQ1)

### 3.1 Interventional vs. Conditional Replacement (Extension B3)
Evaluated across 13 datasets in `results/extensions_v1/conditional/` (seeds 40–49):

| Dataset | Interventional PACE-SHAP Deletion AUC | Conditional PACE-SHAP ($\sigma = 0.1$) Deletion AUC | Deletion $\Delta$ | Share of Inputs Off-Manifold (Interv.) | Share Off-Manifold (Cond.) |
|---|---|---|---|---|---|
| **S1** | **$-0.141$** | $-0.105$ | $+0.036$ (Worse) | $42.5\%$ | **$2.1\%$** |
| **S2** | **$-0.172$** | $-0.128$ | $+0.044$ (Worse) | $38.4\%$ | **$1.8\%$** |
| **Wine** | **$-0.076$** | $-0.052$ | $+0.024$ (Worse) | $58.2\%$ | **$6.4\%$** |
| **Breast Cancer** | **$0.181$** | $0.215$ | $+0.034$ (Worse) | $67.1\%$ | **$8.5\%$** |
| **Thyroid** | **$0.551$** | $0.582$ | $+0.031$ (Worse) | $24.5\%$ | **$1.2\%$** |
| **Shuttle** | **$0.328$** | $0.345$ | $+0.017$ (Worse) | $15.2\%$ | **$0.0\%$** |

#### Causal Root Cause
* **The Manifold Trade-Off:** The conditional game succeeds at keeping synthetic inputs on-manifold (dropping off-manifold inputs from up to $67\%$ down to $< 10\%$).
* **Effective Sample Size Collapse:** However, at narrow bandwidth $\sigma = 0.1$, the Gaussian weights decay so fast that for large coalitions, the conditional game evaluates against approximately **one single background instance**. This extreme variance degrades empirical deletion faithfulness compared to interventional averaging.

---

### 3.2 Explainer Variety Ablation (Faithfulness on Margin)
Evaluated across 11 datasets in `results/faithfulness/` (seeds 0–9):

| Explainer Method | Deletion AUC Win Count vs. KernelSHAP | Insertion AUC Win Count vs. KernelSHAP | Mean Latency (ms/point) |
|---|---|---|---|
| **PACE-SHAP** | **6 Wins / 5 Ties / 0 Losses** | **7 Wins / 4 Ties / 0 Losses** | $45.2$ ms (200 rows) |
| **PACE-flip** | 3 Wins / 2 Ties / 6 Losses | 2 Wins / 3 Ties / 6 Losses | **$0.32$ ms** |
| **PACE-multi** | 1 Win / 2 Ties / 8 Losses | 1 Win / 1 Tie / 9 Losses | **$0.18$ ms** |
| **Contrastive** | 0 Wins / 2 Ties / 9 Losses | 0 Wins / 1 Tie / 10 Losses | **$0.08$ ms** |
| **Decomposition**| 0 Wins / 0 Ties / 11 Losses | 0 Wins / 0 Ties / 11 Losses | **$0.05$ ms** |

* *Takeaway:* `contrastive` and `decomposition` explain membership in isolated boxes rather than the overall decision margin, causing them to perform poorly on coalitional deletion tests. `PaceShap` is the only explainer that rigorously answers coalitional margin questions.

---

## 4. Trust Telemetry & Error Detection Ablations (RQ2)

Evaluated in `results/error_detection_confirmatory/` across seeds 10–19:

### Stepwise Signal Addition (AUPRC for Error Detection)

| Stage | Signal Formulation | Mean AUPRC Across 8 Units | $\Delta$ Over Previous Stage | Causal Explanation |
|---|---|---|---|---|
| **Stage 0** | Random Guessing Baseline | $0.118$ | $-$ | Reflects base error rate |
| **Stage 1** | Plain Margin Uncertainty ($1 - (s'_1 - s'_2)$) | $0.384$ | $+0.266$ | Exploits boundary proximity |
| **Stage 2** | Blueprint $U$ (Ambiguity + Insecurity) | $0.362$ | $-0.022$ (Hurts!) | Flawed tie scoring deprioritizes tied points |
| **Stage 3** | **Corrected Uncertainty ($U^*$)** | **$0.435$** | **$+0.073$** | Prioritizes tied points ($[0.5, 1.0]$) per Proposition 1 |
| **Stage 4** | Prediction Disagreement ($PD$) Alone | $0.442$ | $+0.007$ | Directly tests boundary instability |
| **Stage 5** | Composite $PD + U^*$ | **$0.498$** | **$+0.056$** | Combines boundary proximity and tie ambiguity |
| **Stage 6** | Composite with EI ($PRS = PD + U^* + EI$)| $0.485$ | $-0.013$ (No gain) | Contrastive instability adds noise without signal |
| **Stage 7** | **Learned Multimodal Combiner (RQ2 v2)** | **$0.628$** | **$+0.130$** | SOTA: Integrates kNN and order ensemble |

---

## 5. Surgical Self-Repair & Gate Ablations (RQ3 / RQ4)

Evaluated in `results/refinement_v2_confirmatory/` and `results/refinement_v2_explore/`:

### 5.1 Chooser Ablations: How to Select the Feature to Cut

When point $x$ is misclassified inside wrong box $k$, which feature $i$ should be cut?

| Chooser Strategy | Selection Logic | Test Macro-F1 Gain | Accepted Edits / 300 | Rejection Rate (%) |
|---|---|---|---|---|
| **Random (`random`)** | Uniform random feature $i \in \{1, \dots, d\}$ | $+1.2$ to $+3.8$ | $42$ | $86.0\%$ |
| **Volume (`volume`, Default)** | Cut removing smallest width fraction | **$+1.5$ to $+6.0$** | **$88$** | **$70.7\%$** |
| **Fewest Members (`fewest`)** | Cut dropping fewest training points | $+1.4$ to $+5.8$ | $82$ | $72.7\%$ |
| **Contrastive (`explained`)** | Feature with largest contrastive attribution | $+1.3$ to $+5.5$ | $65$ | $78.3\%$ |
| **PACE-SHAP (`pace_shap`)** | Feature with largest PACE-SHAP attribution | $+1.4$ to $+5.7$ | $71$ | $76.3\%$ |
| **Oracle (`oracle`)** | Tries all features through gate; picks best | $+2.1$ to $+7.2$ | $145$ | $51.7\%$ |

* *Primary Finding (Refinement Confirmatory R1):* Cutting the feature chosen by the explanation (`"explained"`) was **never statistically superior** to the minimal geometric slice (`"volume"`). Explanation-guided cuts frequently remove large portions of hyperbox volume, causing the gate to reject the edit.

---

### 5.2 Self-Repair vs. Simpson's Contraction (Run F)

Evaluated in `results/contraction/runs.csv` (seeds 120–129) on 6 datasets:

| Dataset | CFMNN Base | Simpson Contraction (Every Overlap) | Gated Self-Repair (SR Alone) | SR vs. Contraction $\Delta$ |
|---|---|---|---|---|
| **Thyroid** | $84.22\%$ | $81.15\%$ (Severe loss!) | **$90.24\%$** | **$+9.09\%$ Macro-F1** |
| **Shuttle** | $88.45\%$ | $86.40\%$ | **$91.25\%$** | **$+4.85\%$ Macro-F1** |
| **Skin** | $92.15\%$ | $91.50\%$ | **$97.87\%$** | **$+6.37\%$ Macro-F1** |
| **MAGIC** | $72.15\%$ | $68.45\%$ | **$75.45\%$** | **$+7.00\%$ Macro-F1** |
| **Letter** | $72.85\%$ | $69.15\%$ | **$75.15\%$** | **$+6.00\%$ Macro-F1** |
| **Landsat** | $81.25\%$ | $78.50\%$ | **$82.73\%$** | **$+4.23\%$ Macro-F1** |

* *Verdict:* Simpson's classical contraction heuristic carves overlapping boxes aggressively on every training overlap, damaging boundary integrity and reducing macro-F1 across all 6 datasets. Gated Self-Repair acts surgically only on verified training errors, winning on **6 of 6 datasets** (Wilcoxon $p = 0.031$).

---

## 6. Cross-References

- For the base model training pipeline: [`10_Training_Pipeline_and_Optimization_Strategy.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/10_Training_Pipeline_and_Optimization_Strategy.md)
- For the full benchmark results across all baselines: [`13_Overall_Benchmark_Results_and_SOTA_Comparison.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/13_Overall_Benchmark_Results_and_SOTA_Comparison.md)
- For training time and convergence analysis across configurations: [`15_Component_Wise_Training_Time_and_Convergence_Analysis.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/15_Component_Wise_Training_Time_and_Convergence_Analysis.md)
- For statistical significance tests on ablations: [`17_Statistical_Significance_Robustness_and_Stability.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/17_Statistical_Significance_Robustness_and_Stability.md)
- For dedicated deep dive into self-repair vs contraction: [`22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/22_RQ3_Self_Repair_Deep_Dive_and_Certificate_Guarantees.md)
