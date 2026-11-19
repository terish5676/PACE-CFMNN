# Review Test 2 (pre-registered): stability, ties and LIME

Rules: [`../PREREGISTRATION.md`](../PREREGISTRATION.md) (Test 2). Seeds [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]; v3's models, splits and explained points (50 per dataset and seed, 36 on Wine). Deletion: lower is more faithful. Insertion: higher is more faithful.

## Explained points that are tied

| Dataset | Points | Tied (margin 0) | Margin below 0.01 |
|---|---|---|---|
| S1 | 500 | 0% | 6% |
| S2 | 500 | 0% | 10% |
| S3 | 500 | 0% | 7% |
| letter | 500 | 50% | 50% |
| landsat | 500 | 26% | 51% |
| magic | 500 | 77% | 89% |
| wine | 360 | 0% | 1% |
| breast_cancer | 500 | 1% | 8% |
| thyroid | 500 | 4% | 74% |
| shuttle | 500 | 15% | 91% |
| page_blocks | 500 | 70% | 89% |

## S1 (primary): stability when each method keeps its rows

PACE-SHAP minus KernelSHAP in `stability_fixed`; better for PACE-SHAP: positive. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 11 datasets. **PACE-SHAP is better on 11, level on 0 and worse on 0.**

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.0936 ± 0.0218 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.1302 ± 0.0120 | 10 of 10 | 0.00195 | 0.0215 |
| S3 | +0.1286 ± 0.0254 | 10 of 10 | 0.00195 | 0.0215 |
| letter | +0.1344 ± 0.0159 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.7536 ± 0.0491 | 10 of 10 | 0.00195 | 0.0215 |
| magic | +0.1976 ± 0.0293 | 10 of 10 | 0.00195 | 0.0215 |
| wine | +0.0625 ± 0.0226 | 10 of 10 | 0.00195 | 0.0215 |
| breast_cancer | +0.3938 ± 0.0213 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.0416 ± 0.0327 | 8 of 10 | 0.0234 | 0.0469 |
| shuttle | +0.0140 ± 0.0136 | 6 of 10 | 0.0312 | 0.0469 |
| page_blocks | +0.1306 ± 0.0827 | 10 of 10 | 0.00195 | 0.0215 |

## S2 (primary): stability when every method gets new rows

PACE-SHAP minus KernelSHAP in `stability_new`; better for PACE-SHAP: positive. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 11 datasets. **PACE-SHAP is better on 10, level on 1 and worse on 0.**

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.1546 ± 0.0635 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.1944 ± 0.0564 | 10 of 10 | 0.00195 | 0.0215 |
| S3 | +0.3240 ± 0.0664 | 10 of 10 | 0.00195 | 0.0215 |
| letter | +0.2260 ± 0.0740 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.6166 ± 0.0470 | 10 of 10 | 0.00195 | 0.0215 |
| magic | +0.3098 ± 0.1238 | 10 of 10 | 0.00195 | 0.0215 |
| wine | +0.0950 ± 0.0649 | 9 of 10 | 0.00391 | 0.0215 |
| breast_cancer | +0.4066 ± 0.0349 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.3026 ± 0.1451 | 9 of 10 | 0.00391 | 0.0215 |
| shuttle | +0.3162 ± 0.1419 | 9 of 10 | 0.00586 | 0.0215 |
| page_blocks | +0.0582 ± 0.1152 | 6 of 10 | 0.432 | 0.432 |

## F1a: raw deletion margin AUC

PACE-SHAP minus KernelSHAP in `margin_auc_raw`; better for PACE-SHAP: negative. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 11 datasets. **PACE-SHAP is better on 10, level on 1 and worse on 0.**

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.0007 ± 0.0004 | 1 of 10 | 0.00391 | 0.0234 |
| S2 | -0.0013 ± 0.0007 | 2 of 10 | 0.00977 | 0.0234 |
| S3 | -0.0019 ± 0.0007 | 0 of 10 | 0.00195 | 0.0215 |
| letter | -0.0042 ± 0.0004 | 0 of 10 | 0.00195 | 0.0215 |
| landsat | -0.0155 ± 0.0011 | 0 of 10 | 0.00195 | 0.0215 |
| magic | -0.0005 ± 0.0002 | 1 of 10 | 0.00391 | 0.0234 |
| wine | -0.0025 ± 0.0013 | 1 of 10 | 0.00391 | 0.0234 |
| breast_cancer | -0.0104 ± 0.0015 | 0 of 10 | 0.00195 | 0.0215 |
| thyroid | -0.0010 ± 0.0007 | 0 of 10 | 0.00195 | 0.0215 |
| shuttle | -0.0008 ± 0.0007 | 1 of 10 | 0.00391 | 0.0234 |
| page_blocks | -0.0003 ± 0.0004 | 3 of 10 | 0.0645 | 0.0645 |

## F1b: raw insertion margin AUC

PACE-SHAP minus KernelSHAP in `insertion_margin_auc_raw`; better for PACE-SHAP: positive. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 11 datasets. **PACE-SHAP is better on 11, level on 0 and worse on 0.**

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.0015 ± 0.0006 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.0015 ± 0.0008 | 9 of 10 | 0.00586 | 0.0215 |
| S3 | +0.0022 ± 0.0011 | 10 of 10 | 0.00195 | 0.0215 |
| letter | +0.0018 ± 0.0004 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.0112 ± 0.0010 | 10 of 10 | 0.00195 | 0.0215 |
| magic | +0.0005 ± 0.0002 | 10 of 10 | 0.00195 | 0.0215 |
| wine | +0.0028 ± 0.0018 | 8 of 10 | 0.00977 | 0.0215 |
| breast_cancer | +0.0116 ± 0.0010 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.0017 ± 0.0011 | 10 of 10 | 0.00195 | 0.0215 |
| shuttle | +0.0005 ± 0.0005 | 8 of 10 | 0.0195 | 0.0215 |
| page_blocks | +0.0005 ± 0.0004 | 10 of 10 | 0.00195 | 0.0215 |

## F2a: deletion margin AUC on untied points (datasets with at least 10 in every seed)

PACE-SHAP minus KernelSHAP in `margin_auc_untied`; better for PACE-SHAP: negative. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 9 datasets. **PACE-SHAP is better on 7, level on 2 and worse on 0.**

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.0071 ± 0.0023 | 0 of 10 | 0.00195 | 0.0176 |
| S2 | -0.0104 ± 0.0064 | 2 of 10 | 0.0137 | 0.041 |
| S3 | -0.0132 ± 0.0055 | 0 of 10 | 0.00195 | 0.0176 |
| letter | -0.0202 ± 0.0089 | 0 of 10 | 0.00195 | 0.0176 |
| landsat | -0.1236 ± 0.0169 | 0 of 10 | 0.00195 | 0.0176 |
| wine | -0.0065 ± 0.0056 | 3 of 10 | 0.0645 | 0.129 |
| breast_cancer | -0.0603 ± 0.0083 | 0 of 10 | 0.00195 | 0.0176 |
| thyroid | -0.0423 ± 0.0145 | 1 of 10 | 0.00391 | 0.0176 |
| shuttle | +0.0007 ± 0.0122 | 7 of 10 | 0.625 | 0.625 |

## F2b: insertion margin AUC on untied points (datasets with at least 10 in every seed)

PACE-SHAP minus KernelSHAP in `insertion_margin_auc_untied`; better for PACE-SHAP: positive. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 9 datasets. **PACE-SHAP is better on 6, level on 3 and worse on 0.**

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.0088 ± 0.0052 | 9 of 10 | 0.00586 | 0.0234 |
| S2 | +0.0088 ± 0.0070 | 8 of 10 | 0.0273 | 0.082 |
| S3 | +0.0138 ± 0.0082 | 9 of 10 | 0.00391 | 0.0195 |
| letter | +0.0319 ± 0.0117 | 10 of 10 | 0.00195 | 0.0176 |
| landsat | +0.1376 ± 0.0138 | 10 of 10 | 0.00195 | 0.0176 |
| wine | +0.0099 ± 0.0075 | 7 of 10 | 0.0273 | 0.082 |
| breast_cancer | +0.0626 ± 0.0108 | 10 of 10 | 0.00195 | 0.0176 |
| thyroid | +0.0292 ± 0.0135 | 10 of 10 | 0.00195 | 0.0176 |
| shuttle | +0.0063 ± 0.0103 | 5 of 10 | 0.322 | 0.322 |

## F3a: PACE-SHAP vs LIME on the explainer rows, deletion margin AUC

PACE-SHAP minus LIME in `margin_auc`; better for PACE-SHAP: negative. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 11 datasets. **PACE-SHAP is better on 9, level on 2 and worse on 0.**

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.0326 ± 0.0062 | 0 of 10 | 0.00195 | 0.0215 |
| S2 | -0.0392 ± 0.0065 | 0 of 10 | 0.00195 | 0.0215 |
| S3 | -0.0665 ± 0.0113 | 0 of 10 | 0.00195 | 0.0215 |
| letter | -0.0416 ± 0.0075 | 0 of 10 | 0.00195 | 0.0215 |
| landsat | -0.0368 ± 0.0100 | 0 of 10 | 0.00195 | 0.0215 |
| magic | -0.0487 ± 0.0167 | 0 of 10 | 0.00195 | 0.0215 |
| wine | -0.0403 ± 0.0040 | 0 of 10 | 0.00195 | 0.0215 |
| breast_cancer | -0.0238 ± 0.0053 | 0 of 10 | 0.00195 | 0.0215 |
| thyroid | -0.0416 ± 0.0126 | 0 of 10 | 0.00195 | 0.0215 |
| shuttle | -0.0172 ± 0.0147 | 2 of 10 | 0.0273 | 0.0547 |
| page_blocks | -0.0111 ± 0.0197 | 3 of 10 | 0.232 | 0.232 |

## F3b: PACE-SHAP vs LIME on the explainer rows, insertion margin AUC

PACE-SHAP minus LIME in `insertion_margin_auc`; better for PACE-SHAP: positive. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 11 datasets. **PACE-SHAP is better on 11, level on 0 and worse on 0.**

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.0386 ± 0.0086 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.0420 ± 0.0073 | 10 of 10 | 0.00195 | 0.0215 |
| S3 | +0.0813 ± 0.0060 | 10 of 10 | 0.00195 | 0.0215 |
| letter | +0.1341 ± 0.0162 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.1003 ± 0.0083 | 10 of 10 | 0.00195 | 0.0215 |
| magic | +0.0597 ± 0.0223 | 10 of 10 | 0.00195 | 0.0215 |
| wine | +0.0435 ± 0.0076 | 10 of 10 | 0.00195 | 0.0215 |
| breast_cancer | +0.0212 ± 0.0090 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.0478 ± 0.0183 | 10 of 10 | 0.00195 | 0.0215 |
| shuttle | +0.0350 ± 0.0157 | 9 of 10 | 0.00391 | 0.0215 |
| page_blocks | +0.0314 ± 0.0262 | 8 of 10 | 0.00977 | 0.0215 |

## Secondary: v3's primary test on the new seeds, deletion margin AUC

PACE-SHAP minus KernelSHAP in `margin_auc`; better for PACE-SHAP: negative. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 11 datasets. **PACE-SHAP is better on 6, level on 5 and worse on 0.**

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.0071 ± 0.0023 | 0 of 10 | 0.00195 | 0.0215 |
| S2 | -0.0104 ± 0.0064 | 2 of 10 | 0.0137 | 0.0684 |
| S3 | -0.0132 ± 0.0055 | 0 of 10 | 0.00195 | 0.0215 |
| letter | -0.0205 ± 0.0061 | 0 of 10 | 0.00195 | 0.0215 |
| landsat | -0.1110 ± 0.0155 | 0 of 10 | 0.00195 | 0.0215 |
| magic | -0.0102 ± 0.0104 | 2 of 10 | 0.0488 | 0.195 |
| wine | -0.0065 ± 0.0056 | 3 of 10 | 0.0645 | 0.195 |
| breast_cancer | -0.0603 ± 0.0083 | 0 of 10 | 0.00195 | 0.0215 |
| thyroid | -0.0428 ± 0.0143 | 1 of 10 | 0.00391 | 0.0234 |
| shuttle | +0.0026 ± 0.0127 | 7 of 10 | 0.375 | 0.375 |
| page_blocks | -0.0189 ± 0.0251 | 4 of 10 | 0.131 | 0.262 |

## Secondary: v3's primary test on the new seeds, insertion margin AUC

PACE-SHAP minus KernelSHAP in `insertion_margin_auc`; better for PACE-SHAP: positive. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 11 datasets. **PACE-SHAP is better on 6, level on 5 and worse on 0.**

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.0088 ± 0.0052 | 9 of 10 | 0.00586 | 0.0352 |
| S2 | +0.0088 ± 0.0070 | 8 of 10 | 0.0273 | 0.137 |
| S3 | +0.0138 ± 0.0082 | 9 of 10 | 0.00391 | 0.0273 |
| letter | +0.0436 ± 0.0095 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.1426 ± 0.0142 | 10 of 10 | 0.00195 | 0.0215 |
| magic | +0.0124 ± 0.0110 | 8 of 10 | 0.0488 | 0.146 |
| wine | +0.0099 ± 0.0075 | 7 of 10 | 0.0273 | 0.137 |
| breast_cancer | +0.0622 ± 0.0108 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.0315 ± 0.0123 | 10 of 10 | 0.00195 | 0.0215 |
| shuttle | +0.0076 ± 0.0099 | 6 of 10 | 0.193 | 0.193 |
| page_blocks | +0.0242 ± 0.0292 | 7 of 10 | 0.084 | 0.168 |

## Descriptive: LIME's default discretisation vs v3's setting, deletion margin AUC

LIME-default minus LIME in `margin_auc`; better for LIME-default: negative. Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across the 11 datasets. **LIME-default is better on 1, level on 8 and worse on 2.**

| Dataset | Difference | Seeds where LIME-default is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.0083 ± 0.0090 | 6 of 10 | 0.193 | 0.967 |
| S2 | +0.0064 ± 0.0097 | 7 of 10 | 0.193 | 0.967 |
| S3 | -0.0154 ± 0.0069 | 0 of 10 | 0.00195 | 0.0215 |
| letter | +0.0039 ± 0.0112 | 6 of 10 | 0.557 | 1 |
| landsat | +0.0356 ± 0.0100 | 10 of 10 | 0.00195 | 0.0215 |
| magic | -0.0016 ± 0.0098 | 5 of 10 | 1 | 1 |
| wine | -0.0080 ± 0.0061 | 1 of 10 | 0.0371 | 0.223 |
| breast_cancer | +0.0247 ± 0.0038 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.0077 ± 0.0133 | 6 of 10 | 0.375 | 1 |
| shuttle | +0.0439 ± 0.0272 | 8 of 10 | 0.00977 | 0.0781 |
| page_blocks | +0.0136 ± 0.0080 | 8 of 10 | 0.00977 | 0.0781 |

## Means over seeds (descriptive)

### Stability, rows kept

| Dataset | PACE-SHAP | KernelSHAP | LIME | LIME-default | PACE-flip | Random |
|---|---|---|---|---|---|---|
| S1 | 1.000 | 0.906 | 0.902 | 0.947 | 1.000 | 0.211 |
| S2 | 1.000 | 0.870 | 0.876 | 0.896 | 1.000 | 0.211 |
| S3 | 1.000 | 0.871 | 0.862 | 0.940 | 1.000 | 0.211 |
| letter | 0.976 | 0.842 | 0.843 | 0.728 | 1.000 | 0.125 |
| landsat | 0.932 | 0.178 | 0.508 | 0.422 | 1.000 | 0.053 |
| magic | 1.000 | 0.802 | 0.655 | 0.555 | 1.000 | 0.211 |
| wine | 0.981 | 0.918 | 0.892 | 0.776 | 1.000 | 0.163 |
| breast_cancer | 0.974 | 0.580 | 0.717 | 0.542 | 1.000 | 0.059 |
| thyroid | 0.986 | 0.944 | 0.759 | 0.759 | 1.000 | 0.093 |
| shuttle | 1.000 | 0.986 | 0.848 | 0.615 | 1.000 | 0.223 |
| page_blocks | 1.000 | 0.869 | 0.817 | 0.563 | 1.000 | 0.211 |

### Stability, new rows

| Dataset | PACE-SHAP | KernelSHAP | LIME | LIME-default | PACE-flip | Random |
|---|---|---|---|---|---|---|
| S1 | 0.889 | 0.734 | 0.886 | 0.941 | 0.921 | 0.211 |
| S2 | 0.888 | 0.693 | 0.840 | 0.903 | 0.888 | 0.211 |
| S3 | 0.889 | 0.565 | 0.839 | 0.932 | 0.911 | 0.211 |
| letter | 0.843 | 0.617 | 0.806 | 0.635 | 0.874 | 0.125 |
| landsat | 0.770 | 0.153 | 0.483 | 0.352 | 0.775 | 0.053 |
| magic | 0.762 | 0.452 | 0.625 | 0.518 | 0.838 | 0.211 |
| wine | 0.868 | 0.773 | 0.821 | 0.667 | 0.905 | 0.163 |
| breast_cancer | 0.894 | 0.488 | 0.718 | 0.502 | 0.902 | 0.059 |
| thyroid | 0.769 | 0.466 | 0.694 | 0.630 | 0.835 | 0.093 |
| shuttle | 0.901 | 0.585 | 0.794 | 0.524 | 0.863 | 0.223 |
| page_blocks | 0.532 | 0.474 | 0.634 | 0.458 | 0.739 | 0.211 |

### Deletion margin AUC (normalised, as v3)

| Dataset | PACE-SHAP | KernelSHAP | LIME | LIME-default | PACE-flip | Random |
|---|---|---|---|---|---|---|
| S1 | -0.132 | -0.125 | -0.100 | -0.091 | -0.080 | 0.280 |
| S2 | -0.139 | -0.129 | -0.100 | -0.093 | -0.085 | 0.264 |
| S3 | -0.025 | -0.012 | 0.041 | 0.026 | 0.026 | 0.333 |
| letter | -0.675 | -0.655 | -0.634 | -0.630 | -0.628 | -0.356 |
| landsat | -0.328 | -0.217 | -0.291 | -0.255 | -0.247 | -0.040 |
| magic | -0.049 | -0.039 | -0.000 | -0.002 | 0.019 | 0.126 |
| wine | -0.093 | -0.086 | -0.053 | -0.060 | -0.069 | 0.234 |
| breast_cancer | 0.133 | 0.194 | 0.157 | 0.182 | 0.163 | 0.424 |
| thyroid | 0.531 | 0.574 | 0.573 | 0.580 | 0.534 | 0.738 |
| shuttle | 0.351 | 0.349 | 0.368 | 0.412 | 0.364 | 0.514 |
| page_blocks | 0.061 | 0.080 | 0.073 | 0.086 | 0.079 | 0.200 |

### Insertion margin AUC (normalised, as v3)

| Dataset | PACE-SHAP | KernelSHAP | LIME | LIME-default | PACE-flip | Random |
|---|---|---|---|---|---|---|
| S1 | 0.658 | 0.649 | 0.619 | 0.616 | 0.630 | 0.296 |
| S2 | 0.615 | 0.607 | 0.573 | 0.566 | 0.573 | 0.262 |
| S3 | 0.657 | 0.643 | 0.575 | 0.612 | 0.617 | 0.326 |
| letter | 0.055 | 0.012 | -0.079 | -0.070 | -0.030 | -0.385 |
| landsat | 0.223 | 0.080 | 0.123 | 0.120 | 0.119 | -0.059 |
| magic | 0.262 | 0.249 | 0.202 | 0.199 | 0.211 | 0.087 |
| wine | 0.540 | 0.530 | 0.497 | 0.501 | 0.494 | 0.222 |
| breast_cancer | 0.698 | 0.635 | 0.676 | 0.646 | 0.692 | 0.424 |
| thyroid | 0.912 | 0.881 | 0.865 | 0.875 | 0.915 | 0.725 |
| shuttle | 0.716 | 0.708 | 0.681 | 0.637 | 0.707 | 0.514 |
| page_blocks | 0.317 | 0.293 | 0.285 | 0.265 | 0.305 | 0.138 |

### Deletion margin AUC (raw)

| Dataset | PACE-SHAP | KernelSHAP | LIME | LIME-default | PACE-flip | Random |
|---|---|---|---|---|---|---|
| S1 | -0.020 | -0.019 | -0.016 | -0.015 | -0.012 | 0.031 |
| S2 | -0.018 | -0.017 | -0.014 | -0.013 | -0.010 | 0.027 |
| S3 | -0.004 | -0.002 | 0.004 | 0.002 | 0.004 | 0.037 |
| letter | -0.058 | -0.054 | -0.052 | -0.053 | -0.048 | -0.028 |
| landsat | -0.040 | -0.024 | -0.036 | -0.033 | -0.024 | -0.009 |
| magic | -0.002 | -0.002 | -0.000 | -0.000 | -0.000 | 0.004 |
| wine | -0.040 | -0.038 | -0.029 | -0.031 | -0.028 | 0.042 |
| breast_cancer | 0.011 | 0.021 | 0.015 | 0.018 | 0.015 | 0.053 |
| thyroid | 0.003 | 0.004 | 0.004 | 0.004 | 0.004 | 0.010 |
| shuttle | -0.018 | -0.017 | -0.015 | -0.014 | -0.017 | -0.008 |
| page_blocks | -0.004 | -0.003 | -0.002 | -0.002 | -0.002 | 0.000 |

### Insertion margin AUC (raw)

| Dataset | PACE-SHAP | KernelSHAP | LIME | LIME-default | PACE-flip | Random |
|---|---|---|---|---|---|---|
| S1 | 0.083 | 0.082 | 0.077 | 0.076 | 0.077 | 0.033 |
| S2 | 0.072 | 0.070 | 0.065 | 0.064 | 0.064 | 0.027 |
| S3 | 0.080 | 0.078 | 0.069 | 0.072 | 0.073 | 0.037 |
| letter | -0.000 | -0.002 | -0.007 | -0.006 | -0.007 | -0.029 |
| landsat | 0.013 | 0.002 | 0.006 | 0.005 | 0.002 | -0.010 |
| magic | 0.009 | 0.009 | 0.007 | 0.007 | 0.007 | 0.003 |
| wine | 0.120 | 0.117 | 0.108 | 0.110 | 0.108 | 0.041 |
| breast_cancer | 0.095 | 0.083 | 0.090 | 0.087 | 0.092 | 0.050 |
| thyroid | 0.019 | 0.018 | 0.017 | 0.016 | 0.018 | 0.010 |
| shuttle | 0.003 | 0.002 | 0.000 | -0.001 | 0.002 | -0.007 |
| page_blocks | 0.006 | 0.006 | 0.005 | 0.004 | 0.005 | 0.000 |

### Milliseconds per explanation (runs share the CPU)

| Dataset | PACE-SHAP | KernelSHAP | LIME | LIME-default | PACE-flip | Random |
|---|---|---|---|---|---|---|
| S1 | 9408.768 | 4588.715 | 1169.637 | 1140.322 | 3.286 | 0.007 |
| S2 | 9218.055 | 4596.483 | 1142.457 | 1103.877 | 3.386 | 0.006 |
| S3 | 8762.377 | 4509.982 | 1129.291 | 1102.065 | 3.332 | 0.008 |
| letter | 3169.015 | 2902.763 | 753.950 | 772.022 | 4.494 | 0.008 |
| landsat | 3662.888 | 1453.569 | 374.303 | 444.326 | 3.164 | 0.007 |
| magic | 1148.204 | 820.242 | 211.041 | 224.125 | 2.183 | 0.007 |
| wine | 45.586 | 116.388 | 39.314 | 61.300 | 0.329 | 0.008 |
| breast_cancer | 567.416 | 296.372 | 87.035 | 137.221 | 1.199 | 0.005 |
| thyroid | 377.555 | 243.047 | 72.790 | 105.510 | 0.856 | 0.005 |
| shuttle | 39.087 | 53.175 | 22.523 | 37.251 | 0.135 | 0.005 |
| page_blocks | 154.034 | 65.967 | 24.184 | 38.030 | 0.253 | 0.004 |
