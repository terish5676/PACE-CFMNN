# Global feature importance

Paper's 80/20 split, main model (fixed centroid + search); seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; mean ± 95% CI over seeds. Spearman rank agreement with permutation importance (accuracy drop when a feature is shuffled on the test rows) and with mean abs KernelSHAP over 50 test rows. Ground truth = share of the top 3 that are the truly relevant features (S1-S3 only). KernelSHAP explains the predicted class's margin with 20 real background rows (fair setting); seconds were measured with several runs sharing the CPU, so compare them within a dataset only.

| Dataset | Ranking | Agreement with permutation importance | Agreement with mean abs SHAP | Ground truth top 3 | Seconds |
|---|---|---|---|---|---|
| S1 | PACE data-based | 0.65 ± 0.09 | 0.59 ± 0.09 | 1.00 ± 0.00 | 0.733 ± 0.139 |
| S1 | PACE model-only | 0.64 ± 0.09 | 0.66 ± 0.11 | 1.00 ± 0.00 | 0.429 ± 0.118 |
| S1 | PACE-flip, averaged | 0.72 ± 0.13 | 0.71 ± 0.10 | 1.00 ± 0.00 | 2.022 ± 0.439 |
| S1 | Permutation importance | 1 (reference) | 0.71 ± 0.09 | 1.00 ± 0.00 | 76.483 ± 9.723 |
| S1 | Mean abs KernelSHAP | 0.71 ± 0.09 | 1 (reference) | 1.00 ± 0.00 | 155.696 ± 9.179 |
| S2 | PACE data-based | 0.62 ± 0.10 | 0.69 ± 0.09 | 1.00 ± 0.00 | 0.862 ± 0.227 |
| S2 | PACE model-only | 0.62 ± 0.07 | 0.58 ± 0.10 | 1.00 ± 0.00 | 0.468 ± 0.074 |
| S2 | PACE-flip, averaged | 0.67 ± 0.13 | 0.66 ± 0.07 | 1.00 ± 0.00 | 1.956 ± 0.272 |
| S2 | Permutation importance | 1 (reference) | 0.62 ± 0.08 | 1.00 ± 0.00 | 83.986 ± 1.662 |
| S2 | Mean abs KernelSHAP | 0.62 ± 0.08 | 1 (reference) | 1.00 ± 0.00 | 175.644 ± 2.801 |
| S3 | PACE data-based | 0.60 ± 0.09 | 0.72 ± 0.07 | 1.00 ± 0.00 | 0.732 ± 0.106 |
| S3 | PACE model-only | 0.72 ± 0.10 | 0.69 ± 0.10 | 1.00 ± 0.00 | 0.468 ± 0.173 |
| S3 | PACE-flip, averaged | 0.77 ± 0.09 | 0.68 ± 0.14 | 1.00 ± 0.00 | 1.933 ± 0.282 |
| S3 | Permutation importance | 1 (reference) | 0.72 ± 0.13 | 1.00 ± 0.00 | 86.589 ± 9.474 |
| S3 | Mean abs KernelSHAP | 0.72 ± 0.13 | 1 (reference) | 1.00 ± 0.00 | 135.297 ± 9.445 |
| wine | PACE data-based | 0.47 ± 0.14 | 0.73 ± 0.07 | – | 0.001 ± 0.000 |
| wine | PACE model-only | 0.21 ± 0.16 | 0.88 ± 0.04 | – | 0.002 ± 0.000 |
| wine | PACE-flip, averaged | 0.53 ± 0.25 | 0.54 ± 0.13 | – | 0.007 ± 0.001 |
| wine | Permutation importance | 1 (reference) | 0.23 ± 0.17 | – | 0.325 ± 0.018 |
| wine | Mean abs KernelSHAP | 0.23 ± 0.17 | 1 (reference) | – | 2.900 ± 0.170 |
| breast_cancer | PACE data-based | 0.47 ± 0.08 | 0.81 ± 0.05 | – | 0.008 ± 0.002 |
| breast_cancer | PACE model-only | 0.67 ± 0.10 | 0.79 ± 0.05 | – | 0.003 ± 0.001 |
| breast_cancer | PACE-flip, averaged | 0.71 ± 0.07 | 0.80 ± 0.06 | – | 0.069 ± 0.008 |
| breast_cancer | Permutation importance | 1 (reference) | 0.60 ± 0.06 | – | 2.449 ± 0.279 |
| breast_cancer | Mean abs KernelSHAP | 0.60 ± 0.06 | 1 (reference) | – | 9.990 ± 1.114 |
| thyroid | PACE data-based | 0.59 ± 0.09 | 0.51 ± 0.10 | – | 0.076 ± 0.013 |
| thyroid | PACE model-only | 0.47 ± 0.10 | 0.36 ± 0.13 | – | 0.002 ± 0.000 |
| thyroid | PACE-flip, averaged | 0.55 ± 0.10 | 0.42 ± 0.14 | – | 0.985 ± 0.151 |
| thyroid | Permutation importance | 1 (reference) | 0.51 ± 0.08 | – | 16.084 ± 1.202 |
| thyroid | Mean abs KernelSHAP | 0.51 ± 0.08 | 1 (reference) | – | 11.714 ± 1.197 |
| shuttle | PACE data-based | 0.54 ± 0.04 | 0.53 ± 0.09 | – | 0.134 ± 0.012 |
| shuttle | PACE model-only | 0.34 ± 0.16 | 0.37 ± 0.17 | – | 0.002 ± 0.000 |
| shuttle | PACE-flip, averaged | 0.97 ± 0.01 | 0.93 ± 0.03 | – | 1.752 ± 0.183 |
| shuttle | Permutation importance | 1 (reference) | 0.92 ± 0.02 | – | 9.301 ± 0.788 |
| shuttle | Mean abs KernelSHAP | 0.92 ± 0.02 | 1 (reference) | – | 2.653 ± 0.309 |
| magic | PACE data-based | 0.34 ± 0.12 | 0.09 ± 0.18 | – | 0.661 ± 0.124 |
| magic | PACE model-only | 0.51 ± 0.15 | 0.41 ± 0.13 | – | 0.017 ± 0.004 |
| magic | PACE-flip, averaged | 0.47 ± 0.17 | 0.12 ± 0.19 | – | 7.282 ± 1.400 |
| magic | Permutation importance | 1 (reference) | 0.68 ± 0.17 | – | 74.932 ± 7.168 |
| magic | Mean abs KernelSHAP | 0.68 ± 0.17 | 1 (reference) | – | 35.447 ± 2.430 |
| letter | PACE data-based | 0.78 ± 0.04 | 0.63 ± 0.06 | – | 2.134 ± 0.353 |
| letter | PACE model-only | 0.80 ± 0.02 | 0.84 ± 0.02 | – | 0.175 ± 0.030 |
| letter | PACE-flip, averaged | 0.95 ± 0.01 | 0.80 ± 0.05 | – | 11.091 ± 1.254 |
| letter | Permutation importance | 1 (reference) | 0.87 ± 0.04 | – | 349.557 ± 24.823 |
| letter | Mean abs KernelSHAP | 0.87 ± 0.04 | 1 (reference) | – | 108.965 ± 7.183 |
| landsat | PACE data-based | 0.28 ± 0.11 | 0.37 ± 0.10 | – | 0.346 ± 0.056 |
| landsat | PACE model-only | 0.57 ± 0.08 | 0.65 ± 0.09 | – | 0.052 ± 0.017 |
| landsat | PACE-flip, averaged | 0.84 ± 0.04 | 0.51 ± 0.08 | – | 2.287 ± 0.212 |
| landsat | Permutation importance | 1 (reference) | 0.49 ± 0.08 | – | 112.908 ± 6.665 |
| landsat | Mean abs KernelSHAP | 0.49 ± 0.08 | 1 (reference) | – | 46.196 ± 7.766 |
| page_blocks | PACE data-based | 0.56 ± 0.20 | 0.64 ± 0.09 | – | 0.015 ± 0.002 |
| page_blocks | PACE model-only | -0.09 ± 0.23 | -0.14 ± 0.18 | – | 0.002 ± 0.001 |
| page_blocks | PACE-flip, averaged | 0.63 ± 0.16 | 0.60 ± 0.10 | – | 0.374 ± 0.041 |
| page_blocks | Permutation importance | 1 (reference) | 0.31 ± 0.21 | – | 1.765 ± 0.193 |
| page_blocks | Mean abs KernelSHAP | 0.31 ± 0.21 | 1 (reference) | – | 3.381 ± 0.297 |

## PACE-flip, averaged, minus mean abs KernelSHAP: agreement with permutation importance

Spearman rho with permutation importance; paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across datasets. Positive = PACE-flip agrees better.

| Dataset | Difference | Seeds where PACE-flip is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.02 ± 0.15 | 5 of 10 | 0.742 | 1 |
| S2 | +0.04 ± 0.17 | 5 of 10 | 0.734 | 1 |
| S3 | +0.06 ± 0.15 | 6 of 10 | 0.496 | 1 |
| wine | +0.30 ± 0.19 | 8 of 10 | 0.0195 | 0.156 |
| breast_cancer | +0.11 ± 0.08 | 8 of 10 | 0.0488 | 0.293 |
| thyroid | +0.04 ± 0.12 | 5 of 10 | 0.492 | 1 |
| shuttle | +0.05 ± 0.02 | 10 of 10 | 0.00195 | 0.0215 |
| magic | -0.21 ± 0.30 | 2 of 10 | 0.16 | 0.801 |
| letter | +0.08 ± 0.04 | 9 of 10 | 0.00391 | 0.0352 |
| landsat | +0.35 ± 0.08 | 10 of 10 | 0.00195 | 0.0215 |
| page_blocks | +0.32 ± 0.21 | 8 of 10 | 0.0195 | 0.156 |
