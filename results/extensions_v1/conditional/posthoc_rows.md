# Post hoc: B3's conditional game at σ = 0.1 with more explainer rows

**Post hoc, not pre-registered.** B3 fixed σ = 0.1 and 200 explainer rows in advance, and its results stand as run. The σ check (`posthoc_sigma.md`) gave the conditional game more rows by widening its weights, which also changes the game. This check keeps σ = 0.1 and B3's weight function (the per-feature scale of B3's 200 rows) and adds rows instead: B3's 200 explainer rows followed by further training rows, never the evaluation rows.

Datasets S1, Shuttle, MAGIC, Page Blocks; seeds [40, 41, 42, 43, 44]; B3's models, 50 test points, evaluation rows and tests. For each number of rows (200, 800), PACE-SHAP-cond and PACE-SHAP are computed on the same rows from one shared table of margins.

Descriptive only: with 5 seeds a two-sided Wilcoxon test cannot reach p < 0.05, so no tests are run. Cells give the mean over seeds.

## Check: 200 rows repeat B3

Largest absolute difference from B3's own results, over all datasets and seeds here.

| Method | Faithfulness and ground truth | Inputs off the data |
|---|---|---|
| PACE-SHAP-cond | 0.0e+00 | 0.0e+00 |
| PACE-SHAP | 0.0e+00 | 0.0e+00 |

## 1. Effective explainer rows of the conditional game

Kish effective rows (Σw)² / Σw², median over points and over coalitions with half the features or more (not all); then the median over seeds.

| Dataset | 200 rows | 800 rows |
|---|---|---|
| S1 | 1.1 | 1.5 |
| Shuttle | 3.4 | 9.6 |
| MAGIC | 2.3 | 5.5 |
| Page Blocks | 14.9 | 54.2 |

Smaller coalitions (1 feature up to just under half), same summary:

| Dataset | 200 rows | 800 rows |
|---|---|---|
| S1 | 1.7 | 3.4 |
| Shuttle | 7.5 | 29.2 |
| MAGIC | 3.9 | 12.9 |
| Page Blocks | 24.0 | 93.5 |

## 2. Inputs off the data (%)

B3's measure, 200 inputs per point. Last column: the explained points themselves.

| Dataset | Conditional, 200 rows | Conditional, 800 rows | Interventional, 200 rows | Interventional, 800 rows | Points |
|---|---|---|---|---|---|
| S1 | 1.3 | 0.8 | 3.7 | 3.8 | 4.8 |
| Shuttle | 57.1 | 55.6 | 70.7 | 70.9 | 6.0 |
| MAGIC | 2.2 | 1.5 | 20.1 | 20.2 | 3.6 |
| Page Blocks | 5.5 | 3.7 | 27.1 | 28.0 | 4.8 |

## 3. Conditional test, deletion (lower = more faithful)

In brackets after PACE-SHAP-cond: seeds where it is more faithful than PACE-SHAP on the same rows. Last column: seeds where PACE-SHAP-cond with 800 rows is more faithful than with 200.

| Dataset | PACE-SHAP-cond, 200 rows | PACE-SHAP-cond, 800 rows | PACE-SHAP, 200 rows | PACE-SHAP, 800 rows | 800 vs 200 rows |
|---|---|---|---|---|---|
| S1 | -0.115 (0/5) | -0.149 (0/5) | -0.176 | -0.177 | 5/5 |
| Shuttle | 0.417 (5/5) | 0.407 (5/5) | 0.455 | 0.446 | 4/5 |
| MAGIC | 0.039 (3/5) | 0.036 (4/5) | 0.041 | 0.039 | 3/5 |
| Page Blocks | 0.135 (2/5) | 0.126 (3/5) | 0.128 | 0.133 | 4/5 |

## 4. Conditional test, insertion (higher = more faithful)

In brackets after PACE-SHAP-cond: seeds where it is more faithful than PACE-SHAP on the same rows. Last column: seeds where PACE-SHAP-cond with 800 rows is more faithful than with 200.

| Dataset | PACE-SHAP-cond, 200 rows | PACE-SHAP-cond, 800 rows | PACE-SHAP, 200 rows | PACE-SHAP, 800 rows | 800 vs 200 rows |
|---|---|---|---|---|---|
| S1 | 0.735 (0/5) | 0.758 (0/5) | 0.771 | 0.775 | 5/5 |
| Shuttle | 0.754 (2/5) | 0.759 (3/5) | 0.755 | 0.756 | 4/5 |
| MAGIC | 0.156 (0/5) | 0.150 (1/5) | 0.174 | 0.176 | 2/5 |
| Page Blocks | 0.276 (2/5) | 0.279 (2/5) | 0.276 | 0.283 | 2/5 |

## 5. Interventional test, deletion (lower = more faithful)

In brackets after PACE-SHAP-cond: seeds where it is more faithful than PACE-SHAP on the same rows. Last column: seeds where PACE-SHAP-cond with 800 rows is more faithful than with 200.

| Dataset | PACE-SHAP-cond, 200 rows | PACE-SHAP-cond, 800 rows | PACE-SHAP, 200 rows | PACE-SHAP, 800 rows | 800 vs 200 rows |
|---|---|---|---|---|---|
| S1 | -0.105 (0/5) | -0.125 (0/5) | -0.143 | -0.145 | 5/5 |
| Shuttle | 0.308 (0/5) | 0.298 (1/5) | 0.279 | 0.281 | 5/5 |
| MAGIC | 0.059 (0/5) | 0.047 (0/5) | -0.057 | -0.062 | 3/5 |
| Page Blocks | 0.056 (0/5) | 0.055 (0/5) | 0.015 | 0.014 | 3/5 |

## 6. Interventional test, insertion (higher = more faithful)

In brackets after PACE-SHAP-cond: seeds where it is more faithful than PACE-SHAP on the same rows. Last column: seeds where PACE-SHAP-cond with 800 rows is more faithful than with 200.

| Dataset | PACE-SHAP-cond, 200 rows | PACE-SHAP-cond, 800 rows | PACE-SHAP, 200 rows | PACE-SHAP, 800 rows | 800 vs 200 rows |
|---|---|---|---|---|---|
| S1 | 0.627 (0/5) | 0.646 (0/5) | 0.663 | 0.665 | 5/5 |
| Shuttle | 0.648 (0/5) | 0.662 (0/5) | 0.688 | 0.689 | 4/5 |
| MAGIC | 0.152 (0/5) | 0.144 (0/5) | 0.269 | 0.268 | 3/5 |
| Page Blocks | 0.245 (1/5) | 0.247 (0/5) | 0.290 | 0.293 | 3/5 |

## 7. Ground truth: share of the true features x0–x2 in the top 3

| Dataset | PACE-SHAP-cond, 200 rows | PACE-SHAP-cond, 800 rows | PACE-SHAP, 200 rows | PACE-SHAP, 800 rows |
|---|---|---|---|---|
| S1 | 0.71 | 0.80 | 0.87 | 0.87 |

## 8. Distance between the two games' attributions

Per point Σ|φ_cond − φ_PACE-SHAP| / Σ|φ_PACE-SHAP| on the same rows, mean over points and seeds.

| Dataset | 200 rows | 800 rows |
|---|---|---|
| S1 | 1.19 | 1.00 |
| Shuttle | 1.23 | 1.05 |
| MAGIC | 1.03 | 0.93 |
| Page Blocks | 1.16 | 1.10 |
