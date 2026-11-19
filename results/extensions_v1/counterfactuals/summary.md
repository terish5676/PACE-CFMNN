# Counterfactuals: PACE, PACE-sparse vs DiCE vs nearest unlike neighbour

Paper's 80/20 split, main model (fixed centroid + search); up to 30 test points per dataset and seed; seeds [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]. Target = the runner-up class. Mean ± 95% CI over seeds of per-seed means. L1, features changed and plausibility are over valid counterfactuals only; lower is better for all three. Plausibility = L1 distance to the nearest training point of the target class. Time = median per counterfactual. Thyroid locks features 0 and 1 (age and sex). DiCE: 5 candidates, its own sparse output, the nearest one the model predicts as the target.

PACE-sparse: the fewest changed features that make the model predict the target, scored exactly from the model's own terms and verified on the model (falls back to PACE when PACE changes fewer features). Times are medians with several runs sharing the CPU, so compare them within a dataset only.

| Dataset | Method | Valid (%) | L1 distance | Features changed | Plausibility | Locked kept (%) | ms | Model calls |
|---|---|---|---|---|---|---|---|---|
| S1 | PACE | 100.0 ± 0.0 | 0.386 ± 0.018 | 1.58 ± 0.11 | 1.162 ± 0.026 | – | 6.98 ± 0.41 | 6.4 ± 0.2 |
| S1 | PACE-sparse | 100.0 ± 0.0 | 0.245 ± 0.029 | 1.03 ± 0.02 | 1.273 ± 0.037 | – | 128.97 ± 4.32 | 21.1 ± 0.4 |
| S1 | DiCE | 100.0 ± 0.0 | 0.410 ± 0.039 | 1.32 ± 0.05 | 1.226 ± 0.039 | – | 6784.34 ± 390.25 | – |
| S1 | Nearest unlike neighbour | 100.0 ± 0.0 | 1.420 ± 0.027 | 10.00 ± 0.00 | 0.000 ± 0.000 | – | 0.41 ± 0.01 | – |
| S2 | PACE | 100.0 ± 0.0 | 0.363 ± 0.039 | 1.56 ± 0.10 | 1.114 ± 0.035 | – | 7.23 ± 0.34 | 6.3 ± 0.1 |
| S2 | PACE-sparse | 100.0 ± 0.0 | 0.217 ± 0.032 | 1.05 ± 0.02 | 1.227 ± 0.019 | – | 129.72 ± 3.53 | 20.9 ± 0.6 |
| S2 | DiCE | 100.0 ± 0.0 | 0.391 ± 0.038 | 1.32 ± 0.06 | 1.194 ± 0.021 | – | 6996.27 ± 586.74 | – |
| S2 | Nearest unlike neighbour | 100.0 ± 0.0 | 1.338 ± 0.028 | 10.00 ± 0.00 | 0.000 ± 0.000 | – | 0.42 ± 0.02 | – |
| S3 | PACE | 100.0 ± 0.0 | 0.370 ± 0.016 | 1.44 ± 0.10 | 1.138 ± 0.031 | – | 6.42 ± 0.38 | 6.1 ± 0.2 |
| S3 | PACE-sparse | 100.0 ± 0.0 | 0.233 ± 0.020 | 1.03 ± 0.03 | 1.243 ± 0.029 | – | 123.54 ± 4.69 | 20.6 ± 0.5 |
| S3 | DiCE | 100.0 ± 0.0 | 0.395 ± 0.032 | 1.36 ± 0.09 | 1.236 ± 0.028 | – | 6700.44 ± 236.71 | – |
| S3 | Nearest unlike neighbour | 100.0 ± 0.0 | 1.389 ± 0.031 | 10.00 ± 0.00 | 0.000 ± 0.000 | – | 0.34 ± 0.01 | – |
| wine | PACE | 100.0 ± 0.0 | 0.701 ± 0.042 | 3.44 ± 0.32 | 1.317 ± 0.037 | – | 4.03 ± 0.19 | 7.8 ± 0.3 |
| wine | PACE-sparse | 100.0 ± 0.0 | 0.593 ± 0.045 | 1.89 ± 0.08 | 1.604 ± 0.039 | – | 43.86 ± 3.44 | 16.2 ± 0.4 |
| wine | DiCE | 100.0 ± 0.0 | 0.804 ± 0.052 | 2.35 ± 0.15 | 1.671 ± 0.034 | – | 560.01 ± 23.44 | – |
| wine | Nearest unlike neighbour | 100.0 ± 0.0 | 1.928 ± 0.050 | 12.68 ± 0.07 | 0.000 ± 0.000 | – | 0.06 ± 0.00 | – |
| breast_cancer | PACE | 100.0 ± 0.0 | 1.426 ± 0.122 | 10.96 ± 0.77 | 1.683 ± 0.059 | – | 9.49 ± 0.53 | 24.4 ± 1.9 |
| breast_cancer | PACE-sparse | 100.0 ± 0.0 | 1.188 ± 0.075 | 3.36 ± 0.19 | 2.605 ± 0.085 | – | 4361.70 ± 1104.57 | 34.5 ± 1.9 |
| breast_cancer | DiCE | 100.0 ± 0.0 | 1.660 ± 0.135 | 4.51 ± 0.38 | 3.014 ± 0.087 | – | 1243.47 ± 113.50 | – |
| breast_cancer | Nearest unlike neighbour | 100.0 ± 0.0 | 2.683 ± 0.102 | 29.99 ± 0.02 | 0.056 ± 0.046 | – | 0.11 ± 0.00 | – |
| thyroid | PACE | 99.3 ± 1.0 | 0.154 ± 0.041 | 1.91 ± 0.14 | 0.249 ± 0.026 | 100.0 ± 0.0 | 3.07 ± 0.28 | 6.7 ± 0.5 |
| thyroid | PACE-sparse | 99.3 ± 1.0 | 0.136 ± 0.035 | 1.48 ± 0.12 | 0.294 ± 0.045 | 100.0 ± 0.0 | 15.84 ± 1.67 | 10.5 ± 0.7 |
| thyroid | DiCE | 97.3 ± 2.5 | 0.291 ± 0.053 | 1.60 ± 0.10 | 0.512 ± 0.072 | 100.0 ± 0.0 | 959.13 ± 18.97 | – |
| thyroid | Nearest unlike neighbour | 100.0 ± 0.0 | 0.473 ± 0.076 | 5.89 ± 0.10 | 0.007 ± 0.007 | 9.3 ± 4.2 | 0.32 ± 0.20 | – |
| shuttle | PACE | 100.0 ± 0.0 | 0.117 ± 0.010 | 3.29 ± 0.21 | 0.154 ± 0.009 | – | 7.58 ± 0.99 | 10.7 ± 0.4 |
| shuttle | PACE-sparse | 100.0 ± 0.0 | 0.103 ± 0.010 | 2.31 ± 0.10 | 0.170 ± 0.007 | – | 12.88 ± 1.43 | 12.3 ± 0.5 |
| shuttle | DiCE | 100.0 ± 0.0 | 0.248 ± 0.031 | 1.37 ± 0.09 | 0.285 ± 0.018 | – | 1045.28 ± 28.11 | – |
| shuttle | Nearest unlike neighbour | 100.0 ± 0.0 | 0.242 ± 0.013 | 7.68 ± 0.15 | 0.008 ± 0.005 | – | 0.42 ± 0.02 | – |
| magic | PACE | 100.0 ± 0.0 | 0.250 ± 0.033 | 2.56 ± 0.22 | 0.255 ± 0.016 | – | 7.85 ± 0.73 | 12.8 ± 0.3 |
| magic | PACE-sparse | 100.0 ± 0.0 | 0.110 ± 0.027 | 1.08 ± 0.04 | 0.279 ± 0.026 | – | 47.35 ± 3.49 | 23.1 ± 0.5 |
| magic | DiCE | 100.0 ± 0.0 | 0.327 ± 0.014 | 1.41 ± 0.08 | 0.457 ± 0.025 | – | 2966.48 ± 159.09 | – |
| magic | Nearest unlike neighbour | 100.0 ± 0.0 | 0.302 ± 0.027 | 10.00 ± 0.01 | 0.162 ± 0.020 | – | 1.34 ± 0.07 | – |
| letter | PACE | 100.0 ± 0.0 | 0.216 ± 0.038 | 2.68 ± 0.32 | 0.725 ± 0.054 | – | 31.23 ± 3.46 | 15.8 ± 0.6 |
| letter | PACE-sparse | 100.0 ± 0.0 | 0.113 ± 0.030 | 1.16 ± 0.07 | 0.759 ± 0.058 | – | 99.10 ± 3.50 | 25.3 ± 0.8 |
| letter | DiCE | 98.7 ± 1.2 | 0.322 ± 0.031 | 1.45 ± 0.06 | 0.920 ± 0.057 | – | 12445.51 ± 4398.15 | – |
| letter | Nearest unlike neighbour | 100.0 ± 0.0 | 0.732 ± 0.069 | 7.31 ± 0.29 | 0.303 ± 0.035 | – | 0.58 ± 0.03 | – |
| landsat | PACE | 100.0 ± 0.0 | 1.014 ± 0.284 | 10.63 ± 1.38 | 1.689 ± 0.084 | – | 24.50 ± 4.09 | 31.6 ± 1.1 |
| landsat | PACE-sparse | 100.0 ± 0.0 | 0.810 ± 0.369 | 2.81 ± 0.71 | 2.193 ± 0.245 | – | 1301.90 ± 839.24 | 40.7 ± 1.3 |
| landsat | DiCE | 99.7 ± 0.8 | 1.100 ± 0.399 | 3.32 ± 1.04 | 2.529 ± 0.267 | – | 3769.64 ± 581.57 | – |
| landsat | Nearest unlike neighbour | 100.0 ± 0.0 | 2.309 ± 0.344 | 30.26 ± 0.71 | 0.492 ± 0.102 | – | 0.47 ± 0.06 | – |
| page_blocks | PACE | 100.0 ± 0.0 | 0.238 ± 0.027 | 2.17 ± 0.21 | 0.234 ± 0.027 | – | 7.96 ± 0.82 | 12.8 ± 0.2 |
| page_blocks | PACE-sparse | 100.0 ± 0.0 | 0.088 ± 0.023 | 1.04 ± 0.03 | 0.285 ± 0.030 | – | 25.20 ± 2.95 | 18.5 ± 0.5 |
| page_blocks | DiCE | 98.7 ± 2.0 | 0.348 ± 0.065 | 1.37 ± 0.10 | 0.512 ± 0.052 | – | 556.03 ± 47.46 | – |
| page_blocks | Nearest unlike neighbour | 100.0 ± 0.0 | 0.253 ± 0.037 | 9.59 ± 0.12 | 0.169 ± 0.014 | – | 0.18 ± 0.03 | – |

## PACE-sparse minus DiCE: features changed (primary)

Per-seed mean difference (points where both are valid), mean ± 95% CI over seeds; two-sided Wilcoxon signed-rank test over seeds, Holm-corrected across datasets. Better for PACE-sparse: lower.

| Dataset | Difference | Seeds where PACE-sparse is lower | Seeds where it is higher | Points | p | p (Holm) |
|---|---|---|---|---|---|---|
| S1 | -0.283 ± 0.047 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| S2 | -0.267 ± 0.062 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| S3 | -0.333 ± 0.072 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| wine | -0.460 ± 0.090 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| breast_cancer | -1.143 ± 0.313 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| thyroid | -0.134 ± 0.077 | 9 of 10 | 1 of 10 | 290 | 0.00391 | 0.0215 |
| shuttle | +0.940 ± 0.111 | 0 of 10 | 10 of 10 | 300 | 0.00195 | 0.0215 |
| magic | -0.330 ± 0.077 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| letter | -0.294 ± 0.070 | 10 of 10 | 0 of 10 | 296 | 0.00195 | 0.0215 |
| landsat | -0.501 ± 0.364 | 9 of 10 | 0 of 10 | 299 | 0.00391 | 0.0215 |
| page_blocks | -0.337 ± 0.106 | 10 of 10 | 0 of 10 | 296 | 0.00195 | 0.0215 |

## PACE-sparse minus DiCE: l1 distance (secondary)

Per-seed mean difference (points where both are valid), mean ± 95% CI over seeds; two-sided Wilcoxon signed-rank test over seeds, Holm-corrected across datasets. Better for PACE-sparse: lower.

| Dataset | Difference | Seeds where PACE-sparse is lower | Seeds where it is higher | Points | p | p (Holm) |
|---|---|---|---|---|---|---|
| S1 | -0.164 ± 0.020 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| S2 | -0.174 ± 0.016 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| S3 | -0.162 ± 0.020 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| wine | -0.211 ± 0.027 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| breast_cancer | -0.472 ± 0.099 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| thyroid | -0.153 ± 0.036 | 10 of 10 | 0 of 10 | 290 | 0.00195 | 0.0215 |
| shuttle | -0.145 ± 0.030 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| magic | -0.217 ± 0.026 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| letter | -0.209 ± 0.010 | 10 of 10 | 0 of 10 | 296 | 0.00195 | 0.0215 |
| landsat | -0.288 ± 0.044 | 10 of 10 | 0 of 10 | 299 | 0.00195 | 0.0215 |
| page_blocks | -0.260 ± 0.055 | 10 of 10 | 0 of 10 | 296 | 0.00195 | 0.0215 |

## PACE-sparse minus DiCE: plausibility (secondary)

Per-seed mean difference (points where both are valid), mean ± 95% CI over seeds; two-sided Wilcoxon signed-rank test over seeds, Holm-corrected across datasets. Better for PACE-sparse: lower.

| Dataset | Difference | Seeds where PACE-sparse is lower | Seeds where it is higher | Points | p | p (Holm) |
|---|---|---|---|---|---|---|
| S1 | +0.047 ± 0.015 | 0 of 10 | 10 of 10 | 300 | 0.00195 | 0.0215 |
| S2 | +0.033 ± 0.023 | 1 of 10 | 9 of 10 | 300 | 0.00586 | 0.0215 |
| S3 | +0.006 ± 0.016 | 3 of 10 | 7 of 10 | 300 | 0.322 | 0.322 |
| wine | -0.068 ± 0.024 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| breast_cancer | -0.408 ± 0.041 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| thyroid | -0.221 ± 0.046 | 10 of 10 | 0 of 10 | 290 | 0.00195 | 0.0215 |
| shuttle | -0.115 ± 0.019 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| magic | -0.178 ± 0.015 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| letter | -0.161 ± 0.026 | 10 of 10 | 0 of 10 | 296 | 0.00195 | 0.0215 |
| landsat | -0.332 ± 0.036 | 10 of 10 | 0 of 10 | 299 | 0.00195 | 0.0215 |
| page_blocks | -0.223 ± 0.060 | 10 of 10 | 0 of 10 | 296 | 0.00195 | 0.0215 |

## PACE-sparse minus PACE: features changed (secondary)

Per-seed mean difference (points where both are valid), mean ± 95% CI over seeds; two-sided Wilcoxon signed-rank test over seeds, Holm-corrected across datasets. Better for PACE-sparse: lower.

| Dataset | Difference | Seeds where PACE-sparse is lower | Seeds where it is higher | Points | p | p (Holm) |
|---|---|---|---|---|---|---|
| S1 | -0.543 ± 0.103 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| S2 | -0.507 ± 0.088 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| S3 | -0.417 ± 0.095 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| wine | -1.553 ± 0.262 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| breast_cancer | -7.593 ± 0.600 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| thyroid | -0.429 ± 0.103 | 10 of 10 | 0 of 10 | 298 | 0.00195 | 0.0215 |
| shuttle | -0.987 ± 0.175 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| magic | -1.480 ± 0.208 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| letter | -1.523 ± 0.293 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| landsat | -7.817 ± 0.772 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| page_blocks | -1.133 ± 0.210 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |

## PACE-sparse minus PACE: l1 distance (secondary)

Per-seed mean difference (points where both are valid), mean ± 95% CI over seeds; two-sided Wilcoxon signed-rank test over seeds, Holm-corrected across datasets. Better for PACE-sparse: lower.

| Dataset | Difference | Seeds where PACE-sparse is lower | Seeds where it is higher | Points | p | p (Holm) |
|---|---|---|---|---|---|---|
| S1 | -0.141 ± 0.028 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| S2 | -0.146 ± 0.021 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| S3 | -0.136 ± 0.015 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| wine | -0.108 ± 0.029 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| breast_cancer | -0.238 ± 0.101 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| thyroid | -0.017 ± 0.014 | 9 of 10 | 1 of 10 | 298 | 0.0195 | 0.0215 |
| shuttle | -0.014 ± 0.005 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| magic | -0.140 ± 0.016 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| letter | -0.103 ± 0.024 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |
| landsat | -0.204 ± 0.108 | 9 of 10 | 1 of 10 | 300 | 0.00586 | 0.0215 |
| page_blocks | -0.150 ± 0.033 | 10 of 10 | 0 of 10 | 300 | 0.00195 | 0.0215 |

## PACE-sparse minus DiCE: valid share (secondary)

Per-seed mean difference (points where both are valid; for validity, the share of valid answers over all points), mean ± 95% CI over seeds; two-sided Wilcoxon signed-rank test over seeds, Holm-corrected across datasets. Better for PACE-sparse: higher.

| Dataset | Difference | Seeds where PACE-sparse is lower | Seeds where it is higher | Points | p | p (Holm) |
|---|---|---|---|---|---|---|
| S1 | +0.000 ± 0.000 | 0 of 10 | 0 of 10 | 300 | 1 | 1 |
| S2 | +0.000 ± 0.000 | 0 of 10 | 0 of 10 | 300 | 1 | 1 |
| S3 | +0.000 ± 0.000 | 0 of 10 | 0 of 10 | 300 | 1 | 1 |
| wine | +0.000 ± 0.000 | 0 of 10 | 0 of 10 | 300 | 1 | 1 |
| breast_cancer | +0.000 ± 0.000 | 0 of 10 | 0 of 10 | 300 | 1 | 1 |
| thyroid | +0.020 ± 0.026 | 0 of 10 | 3 of 10 | 300 | 0.25 | 1 |
| shuttle | +0.000 ± 0.000 | 0 of 10 | 0 of 10 | 300 | 1 | 1 |
| magic | +0.000 ± 0.000 | 0 of 10 | 0 of 10 | 300 | 1 | 1 |
| letter | +0.013 ± 0.012 | 0 of 10 | 4 of 10 | 300 | 0.125 | 1 |
| landsat | +0.003 ± 0.008 | 0 of 10 | 1 of 10 | 300 | 1 | 1 |
| page_blocks | +0.013 ± 0.020 | 0 of 10 | 2 of 10 | 300 | 0.5 | 1 |

## How PACE-sparse found its answer

Share of points: exact subset search, greedy search, PACE's own answer (fewer or equally many changes), none found.

| Dataset | exact | greedy | pace | none |
|---|---|---|---|---|
| S1 | 100% | 0% | 0% | 0% |
| S2 | 100% | 0% | 0% | 0% |
| S3 | 100% | 0% | 0% | 0% |
| wine | 100% | 0% | 0% | 0% |
| breast_cancer | 71% | 29% | 0% | 0% |
| thyroid | 99% | 0% | 0% | 1% |
| shuttle | 100% | 0% | 0% | 0% |
| magic | 100% | 0% | 0% | 0% |
| letter | 100% | 0% | 0% | 0% |
| landsat | 83% | 17% | 0% | 0% |
| page_blocks | 100% | 0% | 0% | 0% |
