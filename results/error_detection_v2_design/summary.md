# RQ2 v2 (exploratory design run): CFMNN's own uncertainty vs data-based trust signals

Seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. Same splits and eligibility (at least 50 errors in every seed) as the RQ2 confirmatory run. Learned combinations: logistic regression on standardised signals, fitted on the inner rows with their labels. Mean ± 95% CI over seeds.

Tested units: the synthetic family (S1, S2, S3, S2-imbalanced, S2-label-noise, S2-shifted) and digits, thyroid, skin, magic, letter, landsat, page_blocks.

## Datasets

| Dataset | Split | Eligible | Accuracy (%) | Errors (min) | Inner errors (min) |
|---|---|---|---|---|---|
| S1 | role split | yes | 88.2 | 207 | 46 |
| S2 | role split | yes | 79.6 | 367 | 93 |
| S3 | role split | yes | 86.1 | 253 | 60 |
| S2-imbalanced | role split | yes | 90.4 | 180 | 35 |
| S2-label-noise | role split | yes | 74.2 | 490 | 116 |
| S2-shifted | role split | yes | 79.6 | 951 | 93 |
| wine | 5-fold CV | no | 96.8 | 3 | 3 |
| breast_cancer | 5-fold CV | no | 95.1 | 17 | 15 |
| digits | 5-fold CV | yes | 95.6 | 72 | 50 |
| thyroid | role split | yes | 96.9 | 78 | 17 |
| shuttle | role split | no | 99.8 | 40 | 10 |
| skin | role split | yes | 95.9 | 294 | 347 |
| magic | role split | yes | 73.6 | 1815 | 474 |
| letter | role split | yes | 73.5 | 2022 | 507 |
| landsat | role split | yes | 84.6 | 349 | 90 |
| page_blocks | role split | yes | 86.6 | 156 | 36 |

## AUPRC (higher is better)

| Unit | Base rate | kNN | Trust Score | Order ensemble | U* | Box evidence | LR: kNN + TS | LR: native | LR: all | LR: all + PD |
|---|---|---|---|---|---|---|---|---|---|---|
| Synthetic family | 0.170 | 0.690 ± 0.009 | 0.548 ± 0.009 | 0.647 ± 0.008 | 0.388 ± 0.006 | 0.227 ± 0.004 | 0.727 ± 0.010 | 0.703 ± 0.007 | **0.764 ± 0.007** | 0.761 ± 0.007 |
| digits | 0.044 | 0.821 ± 0.021 | 0.859 ± 0.018 | 0.591 ± 0.022 | 0.384 ± 0.028 | 0.460 ± 0.035 | **0.872 ± 0.014** | 0.672 ± 0.034 | 0.823 ± 0.041 | 0.806 ± 0.033 |
| thyroid | 0.031 | 0.110 ± 0.010 | 0.113 ± 0.016 | 0.455 ± 0.049 | 0.323 ± 0.064 | 0.065 ± 0.013 | 0.130 ± 0.011 | 0.560 ± 0.071 | **0.560 ± 0.069** | 0.540 ± 0.077 |
| skin | 0.041 | 0.994 ± 0.002 | 0.988 ± 0.008 | 0.785 ± 0.083 | 0.379 ± 0.049 | 0.269 ± 0.116 | 0.996 ± 0.002 | 0.929 ± 0.041 | 0.997 ± 0.002 | **0.997 ± 0.001** |
| magic | 0.264 | 0.698 ± 0.023 | 0.672 ± 0.022 | 0.561 ± 0.030 | 0.407 ± 0.014 | 0.288 ± 0.010 | 0.739 ± 0.020 | 0.621 ± 0.027 | **0.752 ± 0.019** | 0.752 ± 0.019 |
| letter | 0.265 | 0.971 ± 0.002 | 0.972 ± 0.003 | 0.795 ± 0.011 | 0.585 ± 0.011 | 0.666 ± 0.011 | 0.984 ± 0.001 | 0.909 ± 0.006 | **0.986 ± 0.001** | 0.985 ± 0.001 |
| landsat | 0.154 | 0.714 ± 0.019 | 0.725 ± 0.015 | 0.519 ± 0.037 | 0.397 ± 0.033 | 0.268 ± 0.015 | 0.763 ± 0.016 | 0.621 ± 0.030 | **0.771 ± 0.019** | 0.769 ± 0.020 |
| page_blocks | 0.134 | 0.815 ± 0.067 | 0.799 ± 0.072 | 0.511 ± 0.130 | 0.372 ± 0.072 | 0.236 ± 0.078 | **0.858 ± 0.058** | 0.658 ± 0.105 | 0.851 ± 0.056 | 0.846 ± 0.054 |

## AUROC

| Unit | Base rate | kNN | Trust Score | Order ensemble | U* | Box evidence | LR: kNN + TS | LR: native | LR: all | LR: all + PD |
|---|---|---|---|---|---|---|---|---|---|---|
| Synthetic family | 0.170 | 0.909 ± 0.003 | 0.838 ± 0.002 | 0.883 ± 0.003 | 0.783 ± 0.002 | 0.586 ± 0.004 | 0.910 ± 0.003 | 0.897 ± 0.001 | **0.923 ± 0.002** | 0.922 ± 0.002 |
| digits | 0.044 | 0.982 ± 0.005 | **0.991 ± 0.001** | 0.900 ± 0.010 | 0.939 ± 0.003 | 0.937 ± 0.007 | 0.989 ± 0.002 | 0.973 ± 0.003 | 0.989 ± 0.003 | 0.986 ± 0.004 |
| thyroid | 0.031 | 0.749 ± 0.012 | 0.741 ± 0.014 | 0.852 ± 0.035 | 0.736 ± 0.026 | 0.707 ± 0.040 | 0.761 ± 0.015 | **0.942 ± 0.013** | 0.940 ± 0.014 | 0.936 ± 0.018 |
| skin | 0.041 | 0.998 ± 0.001 | 0.999 ± 0.000 | 0.981 ± 0.008 | 0.954 ± 0.010 | 0.917 ± 0.020 | 1.000 ± 0.001 | 0.996 ± 0.002 | **1.000 ± 0.000** | **1.000 ± 0.000** |
| magic | 0.264 | 0.868 ± 0.006 | 0.848 ± 0.006 | 0.803 ± 0.012 | 0.680 ± 0.007 | 0.581 ± 0.013 | 0.881 ± 0.005 | 0.824 ± 0.008 | 0.887 ± 0.004 | **0.887 ± 0.004** |
| letter | 0.265 | 0.991 ± 0.001 | 0.989 ± 0.001 | 0.928 ± 0.005 | 0.828 ± 0.005 | 0.871 ± 0.004 | 0.994 ± 0.000 | 0.966 ± 0.002 | **0.994 ± 0.000** | 0.994 ± 0.000 |
| landsat | 0.154 | 0.931 ± 0.004 | 0.934 ± 0.002 | 0.840 ± 0.010 | 0.802 ± 0.009 | 0.676 ± 0.015 | 0.944 ± 0.002 | 0.902 ± 0.007 | **0.946 ± 0.003** | 0.946 ± 0.003 |
| page_blocks | 0.134 | 0.958 ± 0.013 | 0.951 ± 0.013 | 0.821 ± 0.045 | 0.784 ± 0.028 | 0.738 ± 0.045 | 0.969 ± 0.009 | 0.928 ± 0.020 | 0.968 ± 0.010 | **0.969 ± 0.009** |

## Q1 (draft): do CFMNN-native signals add to the data-based baselines? LR(all) minus LR(kNN + TS) (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.038 ± 0.005 | 10 of 10 | 0.00195 | 0.0156 | 5.52 |
| digits | -0.049 ± 0.035 | 2 of 10 | 0.0195 | 0.0781 | -0.99 |
| thyroid | 0.430 ± 0.073 | 10 of 10 | 0.00195 | 0.0156 | 4.23 |
| skin | 0.001 ± 0.002 | 6 of 10 | 0.432 | 0.432 | 0.31 |
| magic | 0.012 ± 0.003 | 10 of 10 | 0.00195 | 0.0156 | 3.42 |
| letter | 0.002 ± 0.001 | 10 of 10 | 0.00195 | 0.0156 | 2.39 |
| landsat | 0.008 ± 0.007 | 9 of 10 | 0.0195 | 0.0781 | 0.86 |
| page_blocks | -0.007 ± 0.012 | 2 of 10 | 0.131 | 0.262 | -0.41 |

## Q2a: LR(all) minus kNN disagreement (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.074 ± 0.005 | 10 of 10 | 0.00195 | 0.0156 | 11.33 |
| digits | 0.002 ± 0.039 | 5 of 10 | 1 | 1 | 0.05 |
| thyroid | 0.450 ± 0.076 | 10 of 10 | 0.00195 | 0.0156 | 4.23 |
| skin | 0.002 ± 0.002 | 8 of 10 | 0.0371 | 0.0742 | 0.87 |
| magic | 0.054 ± 0.005 | 10 of 10 | 0.00195 | 0.0156 | 7.59 |
| letter | 0.015 ± 0.001 | 10 of 10 | 0.00195 | 0.0156 | 7.44 |
| landsat | 0.058 ± 0.008 | 10 of 10 | 0.00195 | 0.0156 | 5.50 |
| page_blocks | 0.036 ± 0.015 | 10 of 10 | 0.00195 | 0.0156 | 1.68 |

## Q2b: LR(all) minus Trust Score (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.216 ± 0.006 | 10 of 10 | 0.00195 | 0.0156 | 25.30 |
| digits | -0.035 ± 0.034 | 2 of 10 | 0.0371 | 0.0371 | -0.75 |
| thyroid | 0.447 ± 0.066 | 10 of 10 | 0.00195 | 0.0156 | 4.84 |
| skin | 0.009 ± 0.007 | 10 of 10 | 0.00195 | 0.0156 | 1.02 |
| magic | 0.080 ± 0.006 | 10 of 10 | 0.00195 | 0.0156 | 9.41 |
| letter | 0.014 ± 0.003 | 10 of 10 | 0.00195 | 0.0156 | 3.46 |
| landsat | 0.047 ± 0.010 | 10 of 10 | 0.00195 | 0.0156 | 3.47 |
| page_blocks | 0.052 ± 0.021 | 10 of 10 | 0.00195 | 0.0156 | 1.80 |

## Q3: CFMNN-native signals alone, LR(native) minus kNN disagreement (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | 0.013 ± 0.008 | 9 of 10 | 0.00586 | 0.0156 | 1.14 |
| digits | -0.149 ± 0.036 | 0 of 10 | 0.00195 | 0.0156 | -2.95 |
| thyroid | 0.450 ± 0.077 | 10 of 10 | 0.00195 | 0.0156 | 4.16 |
| skin | -0.065 ± 0.040 | 0 of 10 | 0.00195 | 0.0156 | -1.16 |
| magic | -0.077 ± 0.020 | 0 of 10 | 0.00195 | 0.0156 | -2.68 |
| letter | -0.062 ± 0.006 | 0 of 10 | 0.00195 | 0.0156 | -7.85 |
| landsat | -0.092 ± 0.020 | 0 of 10 | 0.00195 | 0.0156 | -3.37 |
| page_blocks | -0.157 ± 0.069 | 0 of 10 | 0.00195 | 0.0156 | -1.63 |

## Design check: does PD add? LR(all + PD) minus LR(all) (AUPRC)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm-corrected across units.

| Unit | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| Synthetic family | -0.003 ± 0.003 | 1 of 10 | 0.0137 | 0.0957 | -0.95 |
| digits | -0.017 ± 0.018 | 2 of 10 | 0.0488 | 0.244 | -0.68 |
| thyroid | -0.020 ± 0.026 | 2 of 10 | 0.0137 | 0.0957 | -0.56 |
| skin | 0.000 ± 0.001 | 4 of 10 | 0.557 | 0.75 | 0.02 |
| magic | -0.000 ± 0.000 | 3 of 10 | 0.375 | 0.75 | -0.36 |
| letter | -0.000 ± 0.000 | 3 of 10 | 0.232 | 0.697 | -0.46 |
| landsat | -0.002 ± 0.001 | 1 of 10 | 0.00391 | 0.0312 | -1.28 |
| page_blocks | -0.005 ± 0.006 | 3 of 10 | 0.105 | 0.422 | -0.60 |

## Drop-one (descriptive): AUPRC lost when a signal is removed from LR(all)

| Unit | U* | Margin | Tie | Box evidence | Box count | Order ensemble | kNN | Trust Score |
|---|---|---|---|---|---|---|---|---|
| Synthetic family | -0.000 | +0.000 | +0.000 | -0.003 | -0.003 | +0.035 | +0.049 | -0.003 |
| digits | -0.043 | +0.002 | -0.000 | +0.008 | +0.011 | -0.014 | +0.011 | +0.041 |
| thyroid | +0.004 | +0.012 | +0.001 | -0.007 | -0.007 | +0.281 | +0.035 | +0.001 |
| skin | -0.001 | -0.000 | +0.000 | -0.001 | +0.000 | -0.000 | +0.001 | +0.000 |
| magic | +0.001 | -0.001 | +0.000 | -0.000 | -0.000 | +0.010 | +0.035 | +0.011 |
| letter | -0.000 | +0.000 | +0.000 | +0.001 | +0.001 | +0.001 | +0.004 | +0.006 |
| landsat | -0.000 | +0.001 | -0.001 | -0.002 | -0.001 | +0.003 | +0.017 | +0.023 |
| page_blocks | -0.003 | -0.002 | -0.002 | -0.001 | -0.004 | -0.000 | +0.030 | +0.020 |
