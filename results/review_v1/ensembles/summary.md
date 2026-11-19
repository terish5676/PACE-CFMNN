# Review Test 3 (pre-registered): is the order ensemble's gain specific to CFMNN?

Rules: [`../PREREGISTRATION.md`](../PREREGISTRATION.md). Seeds [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]. Same splits and eligibility (at least 50 errors in every seed) as the RQ2 confirmatory run. Learned combinations: logistic regression on standardised signals, fitted on the inner rows with their labels. Mean ± 95% CI over seeds.

Tested units: the synthetic family (S1, S2, S3, S2-imbalanced, S2-label-noise, S2-shifted) and skin, letter, magic, digits, thyroid, landsat, page_blocks.

## Datasets

| Dataset | Split | Eligible | Accuracy (%) | Errors (min) | Inner errors (min) |
|---|---|---|---|---|---|
| skin | role split | yes | 94.8 | 242 | 320 |
| letter | role split | yes | 73.6 | 2008 | 468 |
| magic | role split | yes | 73.4 | 1806 | 453 |
| S2-shifted | role split | yes | 79.6 | 993 | 80 |
| S1 | role split | yes | 88.3 | 226 | 49 |
| S2 | role split | yes | 79.9 | 372 | 80 |
| S3 | role split | yes | 86.3 | 246 | 59 |
| S2-imbalanced | role split | yes | 90.2 | 169 | 40 |
| S2-label-noise | role split | yes | 73.9 | 490 | 112 |
| digits | 5-fold CV | yes | 95.6 | 69 | 51 |
| thyroid | role split | yes | 96.7 | 53 | 18 |
| landsat | role split | yes | 84.8 | 336 | 89 |
| page_blocks | role split | yes | 88.7 | 163 | 37 |

## AUPRC (higher is better)

| Unit | Base rate | kNN | Trust Score | Order ensemble | Bagged ensemble | SGD order ensemble | LR: kNN + TS | LR: kNN + TS + order ens. | LR: kNN + TS + bagged ens. | LR: kNN + TS + order ens. + bagged ens. | LR: kNN + TS + SGD order ens. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Synthetic family | 0.170 | 0.695 ± 0.006 | 0.554 ± 0.006 | 0.649 ± 0.007 | 0.607 ± 0.007 | 0.717 ± 0.016 | 0.731 ± 0.008 | 0.770 ± 0.008 | 0.764 ± 0.008 | 0.782 ± 0.008 | **0.814 ± 0.011** |
| skin | 0.052 | 0.993 ± 0.004 | 0.985 ± 0.016 | 0.760 ± 0.148 | 0.777 ± 0.092 | 0.303 ± 0.100 | 0.995 ± 0.004 | 0.995 ± 0.004 | **0.997 ± 0.002** | 0.996 ± 0.003 | 0.995 ± 0.004 |
| letter | 0.264 | 0.972 ± 0.002 | 0.973 ± 0.002 | 0.796 ± 0.008 | 0.766 ± 0.012 | 0.532 ± 0.021 | 0.985 ± 0.002 | **0.985 ± 0.002** | 0.985 ± 0.002 | 0.985 ± 0.002 | 0.984 ± 0.002 |
| magic | 0.266 | 0.703 ± 0.030 | 0.677 ± 0.029 | 0.561 ± 0.037 | 0.567 ± 0.044 | 0.556 ± 0.039 | 0.743 ± 0.027 | 0.755 ± 0.027 | 0.754 ± 0.027 | **0.758 ± 0.027** | 0.744 ± 0.027 |
| digits | 0.044 | 0.832 ± 0.020 | 0.874 ± 0.020 | 0.615 ± 0.020 | 0.493 ± 0.035 | 0.442 ± 0.031 | **0.877 ± 0.020** | 0.861 ± 0.024 | 0.856 ± 0.024 | 0.848 ± 0.027 | 0.873 ± 0.018 |
| thyroid | 0.033 | 0.096 ± 0.017 | 0.098 ± 0.017 | 0.399 ± 0.041 | 0.418 ± 0.068 | 0.087 ± 0.016 | 0.114 ± 0.018 | 0.421 ± 0.045 | 0.437 ± 0.077 | **0.499 ± 0.068** | 0.115 ± 0.020 |
| landsat | 0.152 | 0.725 ± 0.026 | 0.740 ± 0.023 | 0.531 ± 0.036 | 0.520 ± 0.027 | 0.405 ± 0.036 | 0.779 ± 0.023 | **0.787 ± 0.024** | 0.783 ± 0.023 | 0.785 ± 0.024 | 0.775 ± 0.025 |
| page_blocks | 0.113 | 0.809 ± 0.051 | 0.819 ± 0.061 | 0.523 ± 0.110 | 0.556 ± 0.138 | 0.637 ± 0.070 | **0.865 ± 0.045** | 0.865 ± 0.050 | 0.861 ± 0.049 | 0.860 ± 0.051 | 0.863 ± 0.047 |

## AUROC

| Unit | Base rate | kNN | Trust Score | Order ensemble | Bagged ensemble | SGD order ensemble | LR: kNN + TS | LR: kNN + TS + order ens. | LR: kNN + TS + bagged ens. | LR: kNN + TS + order ens. + bagged ens. | LR: kNN + TS + SGD order ens. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Synthetic family | 0.170 | 0.910 ± 0.002 | 0.839 ± 0.004 | 0.882 ± 0.004 | 0.869 ± 0.003 | 0.905 ± 0.008 | 0.911 ± 0.003 | 0.923 ± 0.003 | 0.922 ± 0.003 | 0.927 ± 0.003 | **0.937 ± 0.004** |
| skin | 0.052 | 0.998 ± 0.002 | 0.999 ± 0.001 | 0.981 ± 0.012 | 0.981 ± 0.010 | 0.864 ± 0.045 | 0.999 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | 0.998 ± 0.002 |
| letter | 0.264 | 0.991 ± 0.001 | 0.990 ± 0.001 | 0.930 ± 0.002 | 0.918 ± 0.005 | 0.819 ± 0.009 | 0.994 ± 0.001 | **0.994 ± 0.001** | 0.994 ± 0.001 | **0.994 ± 0.001** | 0.994 ± 0.001 |
| magic | 0.266 | 0.870 ± 0.007 | 0.847 ± 0.007 | 0.801 ± 0.012 | 0.802 ± 0.013 | 0.788 ± 0.017 | 0.881 ± 0.006 | 0.887 ± 0.006 | 0.886 ± 0.006 | **0.889 ± 0.007** | 0.882 ± 0.007 |
| digits | 0.044 | 0.981 ± 0.005 | **0.992 ± 0.001** | 0.908 ± 0.008 | 0.884 ± 0.011 | 0.926 ± 0.013 | 0.987 ± 0.004 | 0.986 ± 0.004 | 0.985 ± 0.005 | 0.983 ± 0.005 | 0.986 ± 0.004 |
| thyroid | 0.033 | 0.720 ± 0.016 | 0.715 ± 0.025 | 0.815 ± 0.018 | 0.840 ± 0.040 | 0.671 ± 0.032 | 0.738 ± 0.022 | 0.862 ± 0.027 | 0.859 ± 0.035 | **0.896 ± 0.026** | 0.734 ± 0.024 |
| landsat | 0.152 | 0.933 ± 0.005 | 0.940 ± 0.003 | 0.845 ± 0.012 | 0.837 ± 0.012 | 0.836 ± 0.018 | 0.949 ± 0.003 | **0.951 ± 0.004** | 0.950 ± 0.003 | 0.950 ± 0.004 | 0.948 ± 0.003 |
| page_blocks | 0.113 | 0.957 ± 0.010 | 0.957 ± 0.011 | 0.836 ± 0.029 | 0.860 ± 0.043 | 0.897 ± 0.019 | 0.970 ± 0.009 | 0.971 ± 0.009 | 0.972 ± 0.007 | 0.973 ± 0.007 | **0.974 ± 0.007** |

## T3a (primary): CFMNN's order ensemble vs a generic bagged ensemble of CFMNNs, each added to kNN + TS (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.006 ± 0.004 | 9 of 10 | 0.0488 | 0.342 | 0.97 |
| skin | -0.002 ± 0.002 | 4 of 10 | 0.275 | 1 | -0.54 |
| letter | 0.000 ± 0.000 | 9 of 10 | 0.0137 | 0.109 | 1.07 |
| magic | 0.001 ± 0.003 | 7 of 10 | 0.432 | 1 | 0.25 |
| digits | 0.005 ± 0.008 | 6 of 10 | 0.232 | 1 | 0.40 |
| thyroid | -0.016 ± 0.049 | 4 of 10 | 0.492 | 1 | -0.23 |
| landsat | 0.004 ± 0.004 | 7 of 10 | 0.084 | 0.504 | 0.65 |
| page_blocks | 0.004 ± 0.008 | 6 of 10 | 0.275 | 1 | 0.38 |

## T3b: does the order ensemble add beyond the bagged ensemble? (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.018 ± 0.003 | 10 of 10 | 0.00195 | 0.0156 | 3.86 |
| skin | -0.001 ± 0.001 | 5 of 10 | 0.375 | 0.75 | -0.55 |
| letter | 0.000 ± 0.000 | 9 of 10 | 0.00391 | 0.0234 | 1.43 |
| magic | 0.004 ± 0.002 | 9 of 10 | 0.00586 | 0.0293 | 1.29 |
| digits | -0.008 ± 0.009 | 2 of 10 | 0.0645 | 0.258 | -0.63 |
| thyroid | 0.062 ± 0.024 | 10 of 10 | 0.00195 | 0.0156 | 1.81 |
| landsat | 0.002 ± 0.003 | 7 of 10 | 0.131 | 0.393 | 0.55 |
| page_blocks | -0.001 ± 0.006 | 6 of 10 | 0.922 | 0.922 | -0.12 |

## T3c: does another learner's order ensemble add to kNN + TS? (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.083 ± 0.008 | 10 of 10 | 0.00195 | 0.0156 | 7.33 |
| skin | -0.001 ± 0.001 | 1 of 10 | 0.00391 | 0.0273 | -0.48 |
| letter | -0.000 ± 0.000 | 2 of 10 | 0.0195 | 0.117 | -0.74 |
| magic | 0.001 ± 0.002 | 7 of 10 | 0.375 | 1 | 0.34 |
| digits | -0.003 ± 0.008 | 3 of 10 | 0.322 | 1 | -0.31 |
| thyroid | 0.001 ± 0.006 | 5 of 10 | 1 | 1 | 0.09 |
| landsat | -0.004 ± 0.007 | 3 of 10 | 0.232 | 1 | -0.42 |
| page_blocks | -0.003 ± 0.006 | 4 of 10 | 0.492 | 1 | -0.31 |

## T3d (replication of v2's Q1): does CFMNN's order ensemble add to kNN + TS? (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.040 ± 0.005 | 10 of 10 | 0.00195 | 0.0156 | 5.57 |
| skin | -0.000 ± 0.001 | 5 of 10 | 0.77 | 1 | -0.01 |
| letter | 0.001 ± 0.000 | 9 of 10 | 0.00586 | 0.0234 | 1.31 |
| magic | 0.011 ± 0.003 | 10 of 10 | 0.00195 | 0.0156 | 3.23 |
| digits | -0.016 ± 0.012 | 0 of 10 | 0.00195 | 0.0156 | -0.91 |
| thyroid | 0.307 ± 0.032 | 10 of 10 | 0.00195 | 0.0156 | 6.98 |
| landsat | 0.008 ± 0.005 | 9 of 10 | 0.0137 | 0.041 | 1.15 |
| page_blocks | -0.000 ± 0.012 | 6 of 10 | 0.557 | 1 | -0.01 |
