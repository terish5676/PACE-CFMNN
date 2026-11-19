# RQ2 v2 (confirmatory, pre-registered): CFMNN's own uncertainty vs data-based trust signals

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Seeds [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]. Same splits and eligibility (at least 50 errors in every seed) as the RQ2 confirmatory run. Learned combinations: logistic regression on standardised signals, fitted on the inner rows with their labels. Mean ± 95% CI over seeds.

Tested units: the synthetic family (S1, S2, S3, S2-imbalanced, S2-label-noise, S2-shifted) and digits, thyroid, skin, magic, letter, landsat, page_blocks.

## Datasets

| Dataset | Split | Eligible | Accuracy (%) | Errors (min) | Inner errors (min) |
|---|---|---|---|---|---|
| S1 | role split | yes | 88.1 | 220 | 48 |
| S2 | role split | yes | 79.1 | 379 | 87 |
| S3 | role split | yes | 86.4 | 258 | 54 |
| S2-imbalanced | role split | yes | 90.3 | 181 | 40 |
| S2-label-noise | role split | yes | 74.0 | 488 | 117 |
| S2-shifted | role split | yes | 80.0 | 945 | 87 |
| wine | 5-fold CV | no | 97.4 | 2 | 1 |
| breast_cancer | 5-fold CV | no | 95.6 | 20 | 16 |
| digits | 5-fold CV | yes | 95.4 | 68 | 55 |
| thyroid | role split | yes | 97.0 | 67 | 10 |
| shuttle | role split | no | 99.3 | 37 | 6 |
| skin | role split | yes | 95.2 | 616 | 708 |
| magic | role split | yes | 73.8 | 1786 | 460 |
| letter | role split | yes | 73.8 | 2021 | 497 |
| landsat | role split | yes | 84.6 | 372 | 85 |
| page_blocks | role split | yes | 89.7 | 150 | 45 |

## AUPRC (higher is better)

| Unit | Base rate | kNN | Trust Score | Order ensemble | U* | Box evidence | LR: kNN + TS | LR: kNN + TS + order ens. | LR: native | LR: all |
|---|---|---|---|---|---|---|---|---|---|---|
| Synthetic family | 0.170 | 0.697 ± 0.007 | 0.558 ± 0.011 | 0.652 ± 0.010 | 0.392 ± 0.007 | 0.225 ± 0.006 | 0.733 ± 0.007 | **0.774 ± 0.008** | 0.706 ± 0.008 | 0.771 ± 0.006 |
| digits | 0.046 | 0.834 ± 0.018 | **0.877 ± 0.012** | 0.606 ± 0.049 | 0.418 ± 0.042 | 0.489 ± 0.049 | 0.873 ± 0.015 | 0.853 ± 0.024 | 0.698 ± 0.050 | 0.842 ± 0.033 |
| thyroid | 0.030 | 0.100 ± 0.016 | 0.109 ± 0.025 | 0.429 ± 0.073 | 0.288 ± 0.076 | 0.068 ± 0.011 | 0.118 ± 0.021 | 0.458 ± 0.078 | 0.494 ± 0.089 | **0.502 ± 0.118** |
| skin | 0.048 | 0.996 ± 0.002 | 0.994 ± 0.002 | 0.828 ± 0.080 | 0.428 ± 0.048 | 0.280 ± 0.085 | 0.998 ± 0.001 | **0.998 ± 0.001** | 0.949 ± 0.035 | 0.998 ± 0.001 |
| magic | 0.262 | 0.696 ± 0.020 | 0.671 ± 0.018 | 0.545 ± 0.020 | 0.407 ± 0.010 | 0.288 ± 0.012 | 0.738 ± 0.017 | 0.749 ± 0.018 | 0.607 ± 0.019 | **0.749 ± 0.018** |
| letter | 0.262 | 0.969 ± 0.002 | 0.969 ± 0.003 | 0.790 ± 0.008 | 0.578 ± 0.011 | 0.663 ± 0.015 | 0.982 ± 0.002 | 0.983 ± 0.002 | 0.908 ± 0.006 | **0.984 ± 0.002** |
| landsat | 0.154 | 0.729 ± 0.018 | 0.736 ± 0.016 | 0.535 ± 0.025 | 0.387 ± 0.015 | 0.260 ± 0.012 | 0.777 ± 0.015 | **0.787 ± 0.012** | 0.619 ± 0.028 | 0.783 ± 0.015 |
| page_blocks | 0.103 | 0.816 ± 0.042 | 0.807 ± 0.057 | 0.474 ± 0.090 | 0.344 ± 0.035 | 0.220 ± 0.043 | 0.866 ± 0.041 | **0.870 ± 0.040** | 0.634 ± 0.098 | 0.858 ± 0.036 |

## AUROC

| Unit | Base rate | kNN | Trust Score | Order ensemble | U* | Box evidence | LR: kNN + TS | LR: kNN + TS + order ens. | LR: native | LR: all |
|---|---|---|---|---|---|---|---|---|---|---|
| Synthetic family | 0.170 | 0.910 ± 0.002 | 0.840 ± 0.004 | 0.885 ± 0.002 | 0.785 ± 0.003 | 0.585 ± 0.005 | 0.912 ± 0.002 | **0.925 ± 0.002** | 0.898 ± 0.002 | 0.924 ± 0.001 |
| digits | 0.046 | 0.976 ± 0.005 | **0.991 ± 0.001** | 0.901 ± 0.015 | 0.939 ± 0.007 | 0.939 ± 0.010 | 0.987 ± 0.003 | 0.984 ± 0.005 | 0.971 ± 0.004 | 0.986 ± 0.003 |
| thyroid | 0.030 | 0.731 ± 0.016 | 0.726 ± 0.023 | 0.857 ± 0.031 | 0.709 ± 0.056 | 0.709 ± 0.036 | 0.742 ± 0.019 | 0.878 ± 0.036 | **0.920 ± 0.022** | 0.909 ± 0.018 |
| skin | 0.048 | 0.999 ± 0.001 | 0.999 ± 0.000 | 0.989 ± 0.004 | 0.957 ± 0.007 | 0.912 ± 0.022 | 1.000 ± 0.000 | **1.000 ± 0.000** | 0.997 ± 0.002 | 1.000 ± 0.000 |
| magic | 0.262 | 0.869 ± 0.006 | 0.848 ± 0.005 | 0.796 ± 0.008 | 0.685 ± 0.007 | 0.582 ± 0.011 | 0.881 ± 0.005 | 0.886 ± 0.005 | 0.819 ± 0.006 | **0.886 ± 0.005** |
| letter | 0.262 | 0.990 ± 0.001 | 0.989 ± 0.001 | 0.928 ± 0.003 | 0.830 ± 0.005 | 0.869 ± 0.004 | 0.993 ± 0.001 | 0.994 ± 0.001 | 0.966 ± 0.002 | **0.994 ± 0.001** |
| landsat | 0.154 | 0.935 ± 0.005 | 0.936 ± 0.003 | 0.846 ± 0.010 | 0.803 ± 0.011 | 0.667 ± 0.011 | 0.948 ± 0.003 | **0.950 ± 0.003** | 0.903 ± 0.007 | 0.949 ± 0.003 |
| page_blocks | 0.103 | 0.962 ± 0.010 | 0.957 ± 0.010 | 0.815 ± 0.037 | 0.779 ± 0.026 | 0.759 ± 0.029 | **0.971 ± 0.009** | 0.971 ± 0.009 | 0.920 ± 0.020 | 0.967 ± 0.009 |

## Q1 (primary): does CFMNN's order sensitivity add to the data-based baselines? LR(kNN + TS + order ens.) minus LR(kNN + TS) (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.041 ± 0.003 | 10 of 10 | 0.00195 | 0.0156 | 11.25 |
| digits | -0.020 ± 0.012 | 2 of 10 | 0.00977 | 0.0293 | -1.20 |
| thyroid | 0.341 ± 0.070 | 10 of 10 | 0.00195 | 0.0156 | 3.48 |
| skin | 0.000 ± 0.001 | 4 of 10 | 0.77 | 0.77 | 0.28 |
| magic | 0.010 ± 0.003 | 10 of 10 | 0.00195 | 0.0156 | 2.90 |
| letter | 0.001 ± 0.000 | 10 of 10 | 0.00195 | 0.0156 | 1.56 |
| landsat | 0.010 ± 0.004 | 10 of 10 | 0.00195 | 0.0156 | 1.56 |
| page_blocks | 0.004 ± 0.004 | 7 of 10 | 0.105 | 0.211 | 0.62 |

## Q1b: do the geometry signals add beyond that? LR(all) minus LR(kNN + TS + order ens.) (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | -0.003 ± 0.003 | 4 of 10 | 0.16 | 0.801 | -0.65 |
| digits | -0.012 ± 0.020 | 5 of 10 | 0.432 | 0.93 | -0.41 |
| thyroid | 0.044 ± 0.068 | 6 of 10 | 0.275 | 0.93 | 0.46 |
| skin | -0.001 ± 0.001 | 4 of 10 | 0.232 | 0.93 | -0.58 |
| magic | 0.000 ± 0.002 | 5 of 10 | 0.695 | 0.93 | 0.20 |
| letter | 0.001 ± 0.000 | 10 of 10 | 0.00195 | 0.0156 | 2.28 |
| landsat | -0.004 ± 0.005 | 2 of 10 | 0.084 | 0.504 | -0.57 |
| page_blocks | -0.012 ± 0.011 | 1 of 10 | 0.0137 | 0.0957 | -0.81 |

## Q1c (the draft's primary): LR(all) minus LR(kNN + TS) (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.038 ± 0.004 | 10 of 10 | 0.00195 | 0.0156 | 7.12 |
| digits | -0.031 ± 0.026 | 1 of 10 | 0.0137 | 0.0547 | -0.86 |
| thyroid | 0.384 ± 0.111 | 10 of 10 | 0.00195 | 0.0156 | 2.49 |
| skin | -0.000 ± 0.001 | 2 of 10 | 0.131 | 0.262 | -0.55 |
| magic | 0.011 ± 0.003 | 10 of 10 | 0.00195 | 0.0156 | 2.62 |
| letter | 0.002 ± 0.001 | 10 of 10 | 0.00195 | 0.0156 | 2.97 |
| landsat | 0.006 ± 0.007 | 9 of 10 | 0.084 | 0.252 | 0.60 |
| page_blocks | -0.009 ± 0.012 | 4 of 10 | 0.131 | 0.262 | -0.52 |

## Q2a: LR(kNN + TS + order ens.) minus kNN disagreement (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.077 ± 0.004 | 10 of 10 | 0.00195 | 0.0156 | 13.65 |
| digits | 0.019 ± 0.014 | 8 of 10 | 0.0137 | 0.0156 | 0.99 |
| thyroid | 0.358 ± 0.068 | 10 of 10 | 0.00195 | 0.0156 | 3.75 |
| skin | 0.003 ± 0.001 | 10 of 10 | 0.00195 | 0.0156 | 1.40 |
| magic | 0.052 ± 0.005 | 10 of 10 | 0.00195 | 0.0156 | 7.38 |
| letter | 0.014 ± 0.001 | 10 of 10 | 0.00195 | 0.0156 | 12.53 |
| landsat | 0.057 ± 0.007 | 10 of 10 | 0.00195 | 0.0156 | 5.73 |
| page_blocks | 0.054 ± 0.013 | 10 of 10 | 0.00195 | 0.0156 | 2.93 |

## Q2b: LR(kNN + TS + order ens.) minus Trust Score (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.215 ± 0.005 | 10 of 10 | 0.00195 | 0.0156 | 30.19 |
| digits | -0.024 ± 0.015 | 2 of 10 | 0.0137 | 0.0156 | -1.16 |
| thyroid | 0.350 ± 0.072 | 10 of 10 | 0.00195 | 0.0156 | 3.46 |
| skin | 0.005 ± 0.002 | 10 of 10 | 0.00195 | 0.0156 | 1.89 |
| magic | 0.078 ± 0.004 | 10 of 10 | 0.00195 | 0.0156 | 13.44 |
| letter | 0.014 ± 0.002 | 10 of 10 | 0.00195 | 0.0156 | 6.22 |
| landsat | 0.051 ± 0.013 | 10 of 10 | 0.00195 | 0.0156 | 2.89 |
| page_blocks | 0.063 ± 0.028 | 10 of 10 | 0.00195 | 0.0156 | 1.61 |

## Q3: CFMNN-native signals alone, LR(native) minus kNN disagreement (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.010 ± 0.007 | 8 of 10 | 0.0195 | 0.0195 | 1.03 |
| digits | -0.136 ± 0.040 | 0 of 10 | 0.00195 | 0.0156 | -2.42 |
| thyroid | 0.394 ± 0.082 | 10 of 10 | 0.00195 | 0.0156 | 3.42 |
| skin | -0.047 ± 0.036 | 0 of 10 | 0.00195 | 0.0156 | -0.94 |
| magic | -0.090 ± 0.012 | 0 of 10 | 0.00195 | 0.0156 | -5.43 |
| letter | -0.061 ± 0.004 | 0 of 10 | 0.00195 | 0.0156 | -9.75 |
| landsat | -0.111 ± 0.028 | 0 of 10 | 0.00195 | 0.0156 | -2.87 |
| page_blocks | -0.182 ± 0.070 | 0 of 10 | 0.00195 | 0.0156 | -1.85 |

## Drop-one (descriptive): AUPRC lost when a signal is removed from LR(all)

| Unit | U* | Margin | Tie | Box evidence | Box count | Order ensemble | kNN | Trust Score |
|---|---|---|---|---|---|---|---|---|
| Synthetic family | -0.000 | +0.000 | +0.000 | -0.003 | -0.002 | +0.034 | +0.050 | -0.001 |
| digits | -0.024 | +0.009 | -0.002 | +0.016 | +0.015 | -0.013 | +0.011 | +0.042 |
| thyroid | +0.007 | +0.004 | +0.017 | -0.006 | +0.007 | +0.234 | +0.028 | -0.003 |
| skin | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 | +0.000 | +0.000 |
| magic | +0.000 | -0.001 | -0.000 | -0.000 | -0.000 | +0.009 | +0.038 | +0.012 |
| letter | -0.000 | +0.000 | -0.000 | +0.001 | +0.001 | +0.000 | +0.005 | +0.006 |
| landsat | -0.000 | +0.000 | -0.001 | -0.000 | +0.000 | +0.006 | +0.023 | +0.020 |
| page_blocks | -0.001 | -0.002 | -0.001 | -0.009 | +0.000 | +0.002 | +0.036 | +0.023 |
