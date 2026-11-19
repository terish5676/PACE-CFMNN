# 11. Hyperparameter Configuration and Sensitivity Analysis

## 1. Master Hyperparameter Manifest

Every component of the PACE-CFMNN architecture is parameterized by deterministic, reproducible hyperparameters. The table below consolidates all system parameters, default settings, explored search spaces, and their physical functions:

| Module / Component | Hyperparameter | Default Value | Search / Tuned Range | Functional Role & Mathematical Impact |
|---|---|---|---|---|
| **CFMNN Core** | `theta` ($\theta$) | `0.3` | $[0.05, 0.60]$ (grid: $0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6$) | Maximum hyperbox size budget: $\sum_i (w_i - v_i) \le d \cdot \theta$ |
| | `gamma` ($\gamma$) | `4.0` | $\{2.0, 4.0, 8.0\}$ | Membership ramp slope: $f(t) = \max(0, 1 - \gamma t)$ |
| | `centroid_rule` | `"fixed"` | `{"fixed", "original"}` | Accumulation rule: updates member count and sum for all absorbed points |
| | `expansion` | `"search"` | `{"search", "best"}` | Box growth: scans candidate same-class boxes vs best only |
| | `tie_tol` ($\tau$) | `1e-9` | Fixed ($10^{-9}$) | Numerical tolerance for membership equality in top group $T(x)$ |
| | `chunk_elems` | `65_536` | $[16_384, 262_144]$ | Working memory array size for CPU cache-blocked inference |
| **PACE-SHAP** | `exact_max_d` | `12` | $\{8, 10, 12, 13\}$ | Dimensionality threshold for exact $2^d$ subset enumeration |
| | `n_permutations`| `32` | $\{16, 32, 64, 128\}$ | Antithetic permutation pairs sampled per background row when $d > 12$ |
| | `bg_rows` ($R$) | `200` (71 Wine)| $\{20, 100, 200, 800\}$ | Number of background reference instances for interventional game |
| | `sigma` ($\sigma$) | `None` (Interv.)| $\{0.05, 0.10, 0.20, 0.40, 1.0, \infty\}$| Gaussian kernel bandwidth for conditional game distance weights |
| | `exact_algo` | `"matmul"` | `{"matmul", "recurrence"}` | Linear algebra multiplication vs dynamic recurrence engine |
| **PACE-flip** | `n_quantiles` | `32` | $\{16, 32, 64, \text{None}\}$ | Mid-quantiles evaluated per feature in single-feature occlusion |
| **Trust Signals** | `n_perturbations` ($K$)| `50` | $\{20, 50, 100\}$ | Number of local neighbor perturbations for PD and EI |
| | `scale_ratio` | `0.05` | $\{0.02, 0.05, 0.10\}$ | Perturbation standard deviation scaling relative to feature spread |
| | `discrete_thresh`| `10` | Fixed ($10$) | Threshold of unique values for discrete category redrawing |
| | `redraw_prob` | `0.10` | $\{0.05, 0.10, 0.20\}$ | Empirical redraw probability for discrete dimensions |
| **Self-Repair & Gate** | `max_f1_drop` | `0.5%` | $\{0.0\%, 0.2\%, 0.5\%, 1.0\%\}$ | Maximum allowable drop in macro-F1 across acceptance rows |
| | `max_recall_drop`| `1.0%` | $\{0.5\%, 1.0\%, 2.0\%\}$ | Maximum allowable drop in any single class recall |
| | `margin` | `1e-3` | Fixed ($10^{-3}$) | Coordinate distance boundary face is moved past misclassified point |
| | `max_edits` | `300` (273 PB) | $\{100, 300, 1000, \text{all}\}$ | Maximum number of training error repair edits attempted |
| **Counterfactual** | `steps` | `(0.25, 0.5, 0.75, 1.0)` | Fixed | Interpolation step fractions towards candidate box centroid |
| | `max_boxes` | `10` | $\{5, 10, 20\}$ | Maximum number of target-class candidate boxes evaluated |

---

## 2. Granular Sensitivity Analyses

### 2.1 Hyperbox Size Budget Sweep: $\theta \in [0.05, 0.60]$

The parameter $\theta$ controls the maximum allowable size of any hyperbox. Its impact was evaluated systematically across 13 datasets in `results/readable_baselines/` and `results/repair_theta/`:

```
+----------------------------------------------------------------------------------------------------+
|                                    ACCURACY VS THETA SENSITIVITY                                    |
|                                                                                                    |
|    Accuracy (%)                                                                                    |
|       ^                                                                                            |
|   100 |                 *------------------* Shuttle (Flat across theta)                           |
|       |               *                      * Landsat                                             |
|    80 |             *                          *                                                   |
|       |           *                              * MAGIC                                           |
|    60 |         *                                  * Letter (Massive drop at large theta)          |
|       |       *                                                                                    |
|    40 +-------+--------+--------+--------+--------+--------+-------->                              |
|              0.05     0.10     0.20     0.30     0.40     0.50   Theta                             |
+----------------------------------------------------------------------------------------------------+
```

#### Empirical Observations (`results/repair_theta/summary.md`)
1. **$\theta = 0.30$ vs. $\theta$ Chosen by Validation:**
   - On 7 of 13 datasets (Letter, Landsat, MAGIC, Page Blocks, Digits, S1, S2), tuning $\theta$ via 3-fold cross-validation significantly outperformed the default $\theta = 0.30$.
   - On **Letter**, choosing $\theta = 0.05$ increased accuracy by **$+20.0$ percentage points** over $\theta = 0.30$. At $\theta = 0.30$, hyperboxes are too large, leading to massive cross-class overlaps across 26 letter classes.
   - However, choosing $\theta = 0.05$ requires **$6\times$ to $23\times$ more hyperboxes**, drastically reducing model interpretability.
2. **Interaction with Self-Repair:**
   - At $\theta = 0.30$, Self-Repair raised macro-F1 on 7 datasets without adding a single box.
   - At the tuned $\theta$ (e.g., $0.05$), Self-Repair improved only 3 datasets (by $\le 1.6$ points) because smaller boxes already minimize training overlap.

---

### 2.2 Ramp Sensitivity Sweep: $\gamma \in \{2.0, 4.0, 8.0\}$ (Run I)

The parameter $\gamma$ governs how rapidly membership decays outside a hyperbox. Run I (`results/generality/runs.csv`, seeds 150–159) evaluated $\gamma = 2.0$, $4.0$, and $8.0$ across 13 datasets:

| Dataset | $\gamma = 2.0$ Accuracy | $\gamma = 4.0$ (Default) | $\gamma = 8.0$ Accuracy | Significant Difference? |
|---|---|---|---|---|
| **S1** | $88.62 \pm 0.95$ | $88.74 \pm 0.91$ | $88.58 \pm 0.98$ | None (Holm $p > 0.05$) |
| **Wine** | $97.22 \pm 2.15$ | $97.22 \pm 2.15$ | $97.22 \pm 2.15$ | Identical ($0$ change) |
| **Breast Cancer** | $93.86 \pm 1.42$ | $93.86 \pm 1.42$ | $93.51 \pm 1.50$ | None (Holm $p > 0.05$) |
| **Thyroid** | $97.15 \pm 0.32$ | $97.18 \pm 0.31$ | $97.12 \pm 0.33$ | None (Holm $p > 0.05$) |
| **Shuttle** | $99.80 \pm 0.02$ | $99.80 \pm 0.02$ | $99.80 \pm 0.02$ | Identical ($0$ change) |
| **MAGIC** | $75.82 \pm 0.54$ | $75.64 \pm 0.58$ | $74.91 \pm 0.62$ | $\gamma = 8$ worse ($-0.73\%$) |
| **Letter** | $74.21 \pm 0.48$ | $74.15 \pm 0.45$ | $73.08 \pm 0.51$ | $\gamma = 8$ worse ($-1.07\%$) |

#### Verdict on Ramp Slope $\gamma$
* $\gamma = 4.0$ provides optimal numerical stability.
* Increasing $\gamma$ to $8.0$ causes overly steep membership drop-offs ($1/\gamma = 0.125$), degrading accuracy on complex datasets (MAGIC, Letter).
* Decreasing $\gamma$ to $2.0$ yields equivalent accuracy to $\gamma = 4.0$ on 11 of 13 datasets, confirming architectural robustness.

---

### 2.3 Conditional Kernel Bandwidth: $\sigma \in \{0.05, 0.10, 0.20, 0.40, 1.0, \infty\}$

In conditional PACE-SHAP (`PaceShap(sigma)`), features outside coalition $S$ are weighted by Gaussian proximity: $w_z(S) = \exp\big(-\frac{D_S^2}{2\sigma^2}\big)$. Evaluated in `results/conditional_settings/`:
1. **At $\sigma = 0.10$ (Strict Local Manifold):** Weights decay so aggressively that for large coalitions ($|S| \ge 5$), the effective sample size collapses to approximately **$1$ single background row**. Consequently, estimation variance increases, and PACE-SHAP-cond loses faithfulness compared to interventional PACE-SHAP.
2. **At $\sigma \ge 0.40$ (Broader Kernel):** Effective sample size expands, restoring faithfulness to parity with interventional SHAP on synthetic suites S1 and S2.
3. **As $\sigma \to \infty$:** $w_z(S) \to 1.0$, converging exactly to standard interventional PACE-SHAP.

---

### 2.4 Gate Strictness Sensitivity in Self-Repair

Evaluated in `results/refinement_v2_explore/` across seeds 110–119:

| Gate Configuration | Gate Rules ($\Delta$ F1 / $\Delta$ Recall) | Accepted Edits | Test Macro-F1 Gain | Predictive Churn |
|---|---|---|---|---|
| **No Gate (`gate-none`)** | Accept all boundary shrinks | 300 / 300 ($100\%$) | **$-3.4$ to $-12.1$** (Severe regression!) | High ($> 15\%$) |
| **Loose Gate (`gate-loose`)**| $\Delta \text{F1} \le -1.5\%, \Delta \text{Rec} \le -2.0\%$ | ~180 / 300 ($60\%$) | $+0.8$ to $+3.2$ | Moderate ($4.5\%$) |
| **Default Gate (`default`)**| $\Delta \text{F1} \le -0.5\%, \Delta \text{Rec} \le -1.0\%$ | **60–120 / 300 ($25–40\%$)** | **$+1.5$ to $+6.0$** | Low ($1.2–3.7\%$) |
| **Strict Gate (`gate-strict`)**| $\Delta \text{F1} \le 0.0\%, \Delta \text{Rec} \le 0.0\%$ | 15–40 / 300 ($10\%$) | $+0.4$ to $+1.8$ (Under-repairs) | Ultra-low ($< 0.8\%$) |

#### The Necessity of the Gate
Without the gate (`gate-none`), shrinking boxes to fix training errors causes catastrophic collateral damage to neighboring points, resulting in double-digit accuracy drops. The default gate ($\le 0.5\%$ F1 drop) achieves the ideal pareto frontier between error recovery and non-interference.

---

## 3. Cross-References

- For the mathematical derivation of hyperbox expansion and membership ramps: [`06_Mathematical_Foundations_and_Theoretical_Formulation.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/06_Mathematical_Foundations_and_Theoretical_Formulation.md)
- For the evaluation protocols and baseline configurations: [`12_Evaluation_Protocol_Metrics_and_Baselines.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/12_Evaluation_Protocol_Metrics_and_Baselines.md)
- For the full ablation tables isolating component contributions: [`14_Component_Wise_Accuracy_and_Ablation_Study.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/14_Component_Wise_Accuracy_and_Ablation_Study.md)
- For multi-seed statistical significance and stability analysis: [`17_Statistical_Significance_Robustness_and_Stability.md`](file:///c:/Users/I%20Terish%20charan%20tej/Downloads/pace-cfmnn-main%20(1)/research_docs/17_Statistical_Significance_Robustness_and_Stability.md)
