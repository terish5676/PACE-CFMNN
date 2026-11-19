# Run D: the conditional test at more settings

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). B3's design, seeds 110–119 (10), 50 test points per dataset and seed. Conditional deletion margin AUC: lower is more faithful; conditional insertion margin AUC: higher. Each comparison: paired over seeds, two-sided Wilcoxon signed-rank test, Holm across datasets. Written by `scripts/run_conditional_settings.py`.

## Primary: PACE-SHAP-cond vs KernelSHAP-cond at each setting

**sigma=0.05, rows=200, cond_margin_auc** (negative favours PACE-SHAP-cond): better on 4, level on 2, worse on 0 of 6 datasets.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) | Verdict |
|---|---|---|---|---|---|
| landsat | -0.1146 ± 0.0153 | 0 of 10 | 0.00195 | 0.0117 | better |
| letter | -0.0099 ± 0.0108 | 2 of 10 | 0.105 | 0.129 | level |
| magic | -0.0152 ± 0.0074 | 0 of 10 | 0.00195 | 0.0117 | better |
| shuttle | -0.0215 ± 0.0073 | 0 of 10 | 0.00195 | 0.0117 | better |
| page_blocks | -0.0063 ± 0.0062 | 2 of 10 | 0.0645 | 0.129 | level |
| S1 | -0.0190 ± 0.0059 | 0 of 10 | 0.00195 | 0.0117 | better |

**sigma=0.05, rows=200, cond_insertion_margin_auc** (positive favours PACE-SHAP-cond): better on 4, level on 2, worse on 0 of 6 datasets.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) | Verdict |
|---|---|---|---|---|---|
| landsat | +0.0532 ± 0.0112 | 10 of 10 | 0.00195 | 0.0117 | better |
| letter | +0.0023 ± 0.0120 | 5 of 10 | 1 | 1 | level |
| magic | +0.0102 ± 0.0084 | 7 of 10 | 0.0371 | 0.0742 | level |
| shuttle | +0.0130 ± 0.0063 | 10 of 10 | 0.00195 | 0.0117 | better |
| page_blocks | +0.0120 ± 0.0062 | 10 of 10 | 0.00195 | 0.0117 | better |
| S1 | +0.0133 ± 0.0069 | 10 of 10 | 0.00195 | 0.0117 | better |

**sigma=0.2, rows=200, cond_margin_auc** (negative favours PACE-SHAP-cond): better on 3, level on 3, worse on 0 of 6 datasets.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) | Verdict |
|---|---|---|---|---|---|
| landsat | -0.1774 ± 0.0134 | 0 of 10 | 0.00195 | 0.0117 | better |
| letter | -0.0096 ± 0.0087 | 3 of 10 | 0.0488 | 0.111 | level |
| magic | -0.0104 ± 0.0043 | 0 of 10 | 0.00195 | 0.0117 | better |
| shuttle | -0.0167 ± 0.0084 | 1 of 10 | 0.00391 | 0.0156 | better |
| page_blocks | -0.0054 ± 0.0042 | 3 of 10 | 0.0371 | 0.111 | level |
| S1 | -0.0048 ± 0.0071 | 3 of 10 | 0.193 | 0.193 | level |

**sigma=0.2, rows=200, cond_insertion_margin_auc** (positive favours PACE-SHAP-cond): better on 3, level on 3, worse on 0 of 6 datasets.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) | Verdict |
|---|---|---|---|---|---|
| landsat | +0.0617 ± 0.0135 | 10 of 10 | 0.00195 | 0.0117 | better |
| letter | +0.0069 ± 0.0098 | 6 of 10 | 0.193 | 0.387 | level |
| magic | +0.0046 ± 0.0053 | 7 of 10 | 0.084 | 0.252 | level |
| shuttle | +0.0091 ± 0.0047 | 10 of 10 | 0.00195 | 0.0117 | better |
| page_blocks | +0.0104 ± 0.0069 | 8 of 10 | 0.00977 | 0.0391 | better |
| S1 | +0.0015 ± 0.0038 | 5 of 10 | 0.625 | 0.625 | level |

**sigma=0.4, rows=200, cond_margin_auc** (negative favours PACE-SHAP-cond): better on 3, level on 3, worse on 0 of 6 datasets.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) | Verdict |
|---|---|---|---|---|---|
| landsat | -0.1985 ± 0.0169 | 0 of 10 | 0.00195 | 0.0117 | better |
| letter | -0.0060 ± 0.0055 | 3 of 10 | 0.084 | 0.168 | level |
| magic | -0.0073 ± 0.0036 | 1 of 10 | 0.00391 | 0.0195 | better |
| shuttle | -0.0091 ± 0.0054 | 1 of 10 | 0.00391 | 0.0195 | better |
| page_blocks | -0.0041 ± 0.0038 | 3 of 10 | 0.0488 | 0.146 | level |
| S1 | +0.0008 ± 0.0056 | 3 of 10 | 0.922 | 0.922 | level |

**sigma=0.4, rows=200, cond_insertion_margin_auc** (positive favours PACE-SHAP-cond): better on 3, level on 3, worse on 0 of 6 datasets.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) | Verdict |
|---|---|---|---|---|---|
| landsat | +0.0511 ± 0.0134 | 10 of 10 | 0.00195 | 0.0117 | better |
| letter | +0.0081 ± 0.0048 | 9 of 10 | 0.00391 | 0.0195 | better |
| magic | +0.0067 ± 0.0053 | 7 of 10 | 0.0273 | 0.0586 | level |
| shuttle | +0.0046 ± 0.0033 | 8 of 10 | 0.0195 | 0.0586 | level |
| page_blocks | +0.0054 ± 0.0034 | 8 of 10 | 0.00977 | 0.0391 | better |
| S1 | -0.0010 ± 0.0030 | 4 of 10 | 0.492 | 0.492 | level |

**sigma=0.2, rows=800, cond_margin_auc** (negative favours PACE-SHAP-cond): better on 3, level on 3, worse on 0 of 6 datasets.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) | Verdict |
|---|---|---|---|---|---|
| landsat | -0.1832 ± 0.0123 | 0 of 10 | 0.00195 | 0.0117 | better |
| letter | -0.0090 ± 0.0092 | 2 of 10 | 0.0273 | 0.082 | level |
| magic | -0.0109 ± 0.0050 | 0 of 10 | 0.00195 | 0.0117 | better |
| shuttle | -0.0157 ± 0.0071 | 1 of 10 | 0.00391 | 0.0156 | better |
| page_blocks | -0.0066 ± 0.0051 | 3 of 10 | 0.0488 | 0.0977 | level |
| S1 | -0.0024 ± 0.0046 | 3 of 10 | 0.375 | 0.375 | level |

**sigma=0.2, rows=800, cond_insertion_margin_auc** (positive favours PACE-SHAP-cond): better on 6, level on 0, worse on 0 of 6 datasets.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) | Verdict |
|---|---|---|---|---|---|
| landsat | +0.0638 ± 0.0135 | 10 of 10 | 0.00195 | 0.0117 | better |
| letter | +0.0072 ± 0.0055 | 8 of 10 | 0.0273 | 0.0273 | better |
| magic | +0.0113 ± 0.0039 | 10 of 10 | 0.00195 | 0.0117 | better |
| shuttle | +0.0094 ± 0.0027 | 10 of 10 | 0.00195 | 0.0117 | better |
| page_blocks | +0.0087 ± 0.0058 | 9 of 10 | 0.00391 | 0.0117 | better |
| S1 | +0.0057 ± 0.0031 | 9 of 10 | 0.00391 | 0.0117 | better |

## Effective rows and inputs off the data, by setting (mean over seeds)

Effective rows: Kish (Σw)² / Σw² of the conditional weights, median over points and 100 random coalitions per point, for coalitions with at least half the features (large) and smaller ones. Off the data: share of the game's inputs farther from the training rows than their own nearest-neighbour distances (B3's measure).

| Dataset | Setting | Effective rows (large) | Effective rows (small) | Off the data |
|---|---|---|---|---|
| S1 | sigma=0.05, rows=200 | 1.0 | 1.3 | 0.011 |
| S1 | sigma=0.2, rows=200 | 2.8 | 10.1 | 0.012 |
| S1 | sigma=0.4, rows=200 | 27.4 | 42.5 | 0.021 |
| S1 | sigma=0.2, rows=800 | 5.8 | 36.6 | 0.008 |
| shuttle | sigma=0.05, rows=200 | 1.3 | 4.8 | 0.490 |
| shuttle | sigma=0.2, rows=200 | 17.8 | 41.9 | 0.624 |
| shuttle | sigma=0.4, rows=200 | 83.2 | 100.7 | 0.666 |
| shuttle | sigma=0.2, rows=800 | 68.6 | 170.2 | 0.621 |
| letter | sigma=0.05, rows=200 | 1.0 | 1.0 | 0.101 |
| letter | sigma=0.2, rows=200 | 3.1 | 7.9 | 0.139 |
| letter | sigma=0.4, rows=200 | 34.3 | 48.2 | 0.339 |
| letter | sigma=0.2, rows=800 | 6.7 | 26.8 | 0.099 |
| landsat | sigma=0.05, rows=200 | 1.3 | 1.4 | 0.051 |
| landsat | sigma=0.2, rows=200 | 17.4 | 19.3 | 0.076 |
| landsat | sigma=0.4, rows=200 | 57.6 | 60.8 | 0.228 |
| landsat | sigma=0.2, rows=800 | 67.9 | 76.4 | 0.064 |
| magic | sigma=0.05, rows=200 | 1.0 | 2.3 | 0.035 |
| magic | sigma=0.2, rows=200 | 11.6 | 25.8 | 0.036 |
| magic | sigma=0.4, rows=200 | 65.1 | 77.4 | 0.051 |
| magic | sigma=0.2, rows=800 | 44.1 | 104.0 | 0.029 |
| page_blocks | sigma=0.05, rows=200 | 2.9 | 10.3 | 0.053 |
| page_blocks | sigma=0.2, rows=200 | 49.8 | 70.6 | 0.080 |
| page_blocks | sigma=0.4, rows=200 | 119.8 | 132.3 | 0.148 |
| page_blocks | sigma=0.2, rows=800 | 197.8 | 280.6 | 0.067 |

## Faithfulness by setting and method (mean ± 95% CI over seeds)

| Dataset | Setting | Method | Cond. deletion | Cond. insertion | Interv. deletion | Interv. insertion | ms per explanation |
|---|---|---|---|---|---|---|---|
| landsat | sigma=0.05, rows=200 | PACE-SHAP-cond | 0.259 ± 0.033 | 0.577 ± 0.033 | -0.156 ± 0.040 | 0.037 ± 0.033 | 2064.4 |
| landsat | sigma=0.05, rows=200 | KernelSHAP-cond | 0.373 ± 0.038 | 0.524 ± 0.030 | -0.099 ± 0.039 | -0.010 ± 0.040 | 5489.1 |
| landsat | sigma=0.2, rows=200 | PACE-SHAP-cond | 0.186 ± 0.033 | 0.592 ± 0.029 | -0.191 ± 0.038 | 0.066 ± 0.032 | 1951.3 |
| landsat | sigma=0.2, rows=200 | KernelSHAP-cond | 0.364 ± 0.039 | 0.530 ± 0.029 | -0.110 ± 0.037 | -0.006 ± 0.040 | 5715.6 |
| landsat | sigma=0.4, rows=200 | PACE-SHAP-cond | 0.156 ± 0.031 | 0.582 ± 0.027 | -0.221 ± 0.039 | 0.093 ± 0.030 | 1895.8 |
| landsat | sigma=0.4, rows=200 | KernelSHAP-cond | 0.355 ± 0.037 | 0.531 ± 0.030 | -0.119 ± 0.037 | 0.001 ± 0.041 | 5970.6 |
| landsat | sigma=0.2, rows=800 | PACE-SHAP-cond | 0.182 ± 0.036 | 0.597 ± 0.027 | -0.198 ± 0.041 | 0.071 ± 0.030 | 8292.8 |
| landsat | sigma=0.2, rows=800 | KernelSHAP-cond | 0.365 ± 0.041 | 0.533 ± 0.029 | -0.110 ± 0.039 | -0.003 ± 0.040 | 21964.4 |
| letter | sigma=0.05, rows=200 | PACE-SHAP-cond | -0.511 ± 0.014 | -0.010 ± 0.038 | -0.600 ± 0.013 | -0.190 ± 0.036 | 2586.6 |
| letter | sigma=0.05, rows=200 | KernelSHAP-cond | -0.501 ± 0.013 | -0.012 ± 0.036 | -0.597 ± 0.017 | -0.188 ± 0.039 | 10836.8 |
| letter | sigma=0.2, rows=200 | PACE-SHAP-cond | -0.557 ± 0.012 | 0.038 ± 0.039 | -0.631 ± 0.014 | -0.131 ± 0.040 | 2029.4 |
| letter | sigma=0.2, rows=200 | KernelSHAP-cond | -0.547 ± 0.010 | 0.031 ± 0.036 | -0.624 ± 0.018 | -0.135 ± 0.037 | 10826.5 |
| letter | sigma=0.4, rows=200 | PACE-SHAP-cond | -0.585 ± 0.011 | 0.078 ± 0.035 | -0.662 ± 0.016 | -0.062 ± 0.036 | 2059.9 |
| letter | sigma=0.4, rows=200 | KernelSHAP-cond | -0.579 ± 0.009 | 0.070 ± 0.033 | -0.659 ± 0.017 | -0.070 ± 0.035 | 10662.6 |
| letter | sigma=0.2, rows=800 | PACE-SHAP-cond | -0.575 ± 0.008 | 0.058 ± 0.035 | -0.640 ± 0.013 | -0.109 ± 0.036 | 8632.1 |
| letter | sigma=0.2, rows=800 | KernelSHAP-cond | -0.566 ± 0.011 | 0.050 ± 0.033 | -0.635 ± 0.014 | -0.116 ± 0.034 | 42453.8 |
| magic | sigma=0.05, rows=200 | PACE-SHAP-cond | 0.042 ± 0.014 | 0.159 ± 0.024 | 0.029 ± 0.023 | 0.166 ± 0.025 | 737.7 |
| magic | sigma=0.05, rows=200 | KernelSHAP-cond | 0.057 ± 0.015 | 0.148 ± 0.024 | 0.048 ± 0.021 | 0.149 ± 0.023 | 3825.9 |
| magic | sigma=0.2, rows=200 | PACE-SHAP-cond | 0.030 ± 0.013 | 0.168 ± 0.024 | 0.007 ± 0.027 | 0.183 ± 0.025 | 692.9 |
| magic | sigma=0.2, rows=200 | KernelSHAP-cond | 0.040 ± 0.014 | 0.164 ± 0.023 | 0.023 ± 0.019 | 0.166 ± 0.025 | 3642.1 |
| magic | sigma=0.4, rows=200 | PACE-SHAP-cond | 0.027 ± 0.014 | 0.179 ± 0.024 | -0.016 ± 0.027 | 0.207 ± 0.020 | 670.1 |
| magic | sigma=0.4, rows=200 | KernelSHAP-cond | 0.034 ± 0.015 | 0.173 ± 0.022 | 0.005 ± 0.022 | 0.186 ± 0.023 | 3693.1 |
| magic | sigma=0.2, rows=800 | PACE-SHAP-cond | 0.025 ± 0.016 | 0.175 ± 0.020 | 0.005 ± 0.030 | 0.182 ± 0.021 | 2736.2 |
| magic | sigma=0.2, rows=800 | KernelSHAP-cond | 0.036 ± 0.017 | 0.164 ± 0.020 | 0.025 ± 0.022 | 0.159 ± 0.021 | 12015.9 |
| shuttle | sigma=0.05, rows=200 | PACE-SHAP-cond | 0.492 ± 0.065 | 0.796 ± 0.044 | 0.401 ± 0.064 | 0.702 ± 0.064 | 20.7 |
| shuttle | sigma=0.05, rows=200 | KernelSHAP-cond | 0.513 ± 0.070 | 0.783 ± 0.042 | 0.420 ± 0.066 | 0.683 ± 0.059 | 213.2 |
| shuttle | sigma=0.2, rows=200 | PACE-SHAP-cond | 0.481 ± 0.064 | 0.800 ± 0.042 | 0.385 ± 0.065 | 0.720 ± 0.061 | 19.8 |
| shuttle | sigma=0.2, rows=200 | KernelSHAP-cond | 0.498 ± 0.068 | 0.791 ± 0.041 | 0.401 ± 0.067 | 0.704 ± 0.057 | 212.8 |
| shuttle | sigma=0.4, rows=200 | PACE-SHAP-cond | 0.472 ± 0.063 | 0.805 ± 0.039 | 0.376 ± 0.067 | 0.736 ± 0.056 | 19.5 |
| shuttle | sigma=0.4, rows=200 | KernelSHAP-cond | 0.481 ± 0.066 | 0.800 ± 0.038 | 0.387 ± 0.068 | 0.724 ± 0.054 | 202.6 |
| shuttle | sigma=0.2, rows=800 | PACE-SHAP-cond | 0.482 ± 0.067 | 0.800 ± 0.039 | 0.384 ± 0.068 | 0.724 ± 0.058 | 131.6 |
| shuttle | sigma=0.2, rows=800 | KernelSHAP-cond | 0.498 ± 0.069 | 0.791 ± 0.039 | 0.403 ± 0.067 | 0.706 ± 0.056 | 727.6 |
| page_blocks | sigma=0.05, rows=200 | PACE-SHAP-cond | 0.121 ± 0.065 | 0.268 ± 0.085 | 0.076 ± 0.059 | 0.217 ± 0.068 | 81.0 |
| page_blocks | sigma=0.05, rows=200 | KernelSHAP-cond | 0.128 ± 0.064 | 0.256 ± 0.084 | 0.082 ± 0.058 | 0.207 ± 0.068 | 327.9 |
| page_blocks | sigma=0.2, rows=200 | PACE-SHAP-cond | 0.111 ± 0.061 | 0.284 ± 0.096 | 0.060 ± 0.048 | 0.242 ± 0.077 | 83.0 |
| page_blocks | sigma=0.2, rows=200 | KernelSHAP-cond | 0.116 ± 0.061 | 0.274 ± 0.093 | 0.063 ± 0.049 | 0.234 ± 0.075 | 337.4 |
| page_blocks | sigma=0.4, rows=200 | PACE-SHAP-cond | 0.109 ± 0.061 | 0.283 ± 0.093 | 0.053 ± 0.050 | 0.256 ± 0.078 | 84.7 |
| page_blocks | sigma=0.4, rows=200 | KernelSHAP-cond | 0.113 ± 0.059 | 0.278 ± 0.093 | 0.056 ± 0.049 | 0.252 ± 0.078 | 340.7 |
| page_blocks | sigma=0.2, rows=800 | PACE-SHAP-cond | 0.108 ± 0.062 | 0.288 ± 0.095 | 0.058 ± 0.051 | 0.244 ± 0.080 | 496.9 |
| page_blocks | sigma=0.2, rows=800 | KernelSHAP-cond | 0.115 ± 0.061 | 0.279 ± 0.092 | 0.063 ± 0.050 | 0.239 ± 0.081 | 1284.1 |
| S1 | sigma=0.05, rows=200 | PACE-SHAP-cond | -0.105 ± 0.024 | 0.729 ± 0.013 | -0.069 ± 0.034 | 0.621 ± 0.015 | 5950.2 |
| S1 | sigma=0.05, rows=200 | KernelSHAP-cond | -0.086 ± 0.023 | 0.716 ± 0.015 | -0.055 ± 0.033 | 0.607 ± 0.017 | 13833.9 |
| S1 | sigma=0.2, rows=200 | PACE-SHAP-cond | -0.126 ± 0.022 | 0.745 ± 0.015 | -0.088 ± 0.033 | 0.635 ± 0.016 | 4520.6 |
| S1 | sigma=0.2, rows=200 | KernelSHAP-cond | -0.121 ± 0.022 | 0.743 ± 0.014 | -0.082 ± 0.032 | 0.629 ± 0.014 | 13505.8 |
| S1 | sigma=0.4, rows=200 | PACE-SHAP-cond | -0.142 ± 0.020 | 0.764 ± 0.018 | -0.102 ± 0.030 | 0.649 ± 0.015 | 3870.4 |
| S1 | sigma=0.4, rows=200 | KernelSHAP-cond | -0.143 ± 0.020 | 0.765 ± 0.018 | -0.101 ± 0.031 | 0.649 ± 0.016 | 14498.3 |
| S1 | sigma=0.2, rows=800 | PACE-SHAP-cond | -0.150 ± 0.019 | 0.767 ± 0.015 | -0.101 ± 0.033 | 0.647 ± 0.015 | 19531.4 |
| S1 | sigma=0.2, rows=800 | KernelSHAP-cond | -0.147 ± 0.017 | 0.761 ± 0.015 | -0.096 ± 0.032 | 0.645 ± 0.014 | 40140.8 |

