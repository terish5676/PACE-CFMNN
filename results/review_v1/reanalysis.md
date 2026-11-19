# Re-analysis of existing results for the pre-submission review

Post hoc: written after the review, from results that already existed. Each section answers one of the review's issues (its code is in the heading). Tests are paired over seeds: two-sided Wilcoxon signed-rank, Holm across the datasets, as in the paper. Made by `scripts/review_reanalysis.py`.

## B1. PACE-SHAP vs KernelSHAP on the same 200 rows

From B3 (`extensions_v1/conditional/runs.csv`, seeds 40–49, 13 datasets). There, KernelSHAP is this repository's solver on the same 200 explainer rows as PACE-SHAP, with 200 coalitions and no feature selection. v3's KernelSHAP used 20 of the rows and shap's default feature selection. The tests are the usual interventional deletion and insertion tests.

PACE-SHAP minus KernelSHAP in `margin_auc` (lower is better). **PACE-SHAP is better on 3, level on 10 and worse on 0** (Holm across 13).

| Dataset | PACE-SHAP | KernelSHAP | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE-SHAP | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| S1 | -0.1260 | -0.1247 | -0.0013 | ±0.0021 | ±0.0036 | -0.44 | 5 of 10 | 1 |
| S2 | -0.1424 | -0.1400 | -0.0024 | ±0.0030 | ±0.0052 | -0.57 | 8 of 10 | 0.756 |
| S3 | -0.0525 | -0.0460 | -0.0066 | ±0.0029 (excludes 0) | ±0.0050 (excludes 0) | -1.60 | 10 of 10 | 0.0254 |
| S1c | -0.0634 | -0.0625 | -0.0009 | ±0.0045 | ±0.0077 | -0.14 | 4 of 10 | 1 |
| S2c | -0.1291 | -0.1269 | -0.0022 | ±0.0026 | ±0.0045 | -0.59 | 7 of 10 | 0.756 |
| wine | -0.0724 | -0.0731 | +0.0007 | ±0.0008 | ±0.0014 | +0.60 | 3 of 10 | 1 |
| breast_cancer | 0.1351 | 0.1375 | -0.0024 | ±0.0011 (excludes 0) | ±0.0018 (excludes 0) | -1.63 | 10 of 10 | 0.0254 |
| thyroid | 0.5078 | 0.5077 | +0.0002 | ±0.0013 | ±0.0022 | +0.09 | 6 of 10 | 1 |
| shuttle | 0.3100 | 0.3106 | -0.0006 | ±0.0008 | ±0.0014 | -0.47 | 8 of 10 | 1 |
| magic | -0.0472 | -0.0495 | +0.0023 | ±0.0028 | ±0.0048 | +0.60 | 2 of 10 | 0.488 |
| letter | -0.7027 | -0.7019 | -0.0007 | ±0.0012 | ±0.0020 | -0.43 | 9 of 10 | 0.756 |
| landsat | -0.3298 | -0.2852 | -0.0446 | ±0.0089 (excludes 0) | ±0.0151 (excludes 0) | -3.60 | 10 of 10 | 0.0254 |
| page_blocks | 0.0407 | 0.0411 | -0.0004 | ±0.0026 | ±0.0044 | -0.11 | 5 of 10 | 1 |

PACE-SHAP minus KernelSHAP in `insertion_margin_auc` (higher is better). **PACE-SHAP is better on 2, level on 11 and worse on 0** (Holm across 13).

| Dataset | PACE-SHAP | KernelSHAP | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE-SHAP | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| S1 | 0.6618 | 0.6612 | +0.0005 | ±0.0023 | ±0.0039 | +0.16 | 5 of 10 | 1 |
| S2 | 0.6164 | 0.6128 | +0.0036 | ±0.0033 (excludes 0) | ±0.0057 | +0.77 | 8 of 10 | 0.84 |
| S3 | 0.6513 | 0.6487 | +0.0027 | ±0.0028 | ±0.0048 | +0.68 | 8 of 10 | 0.537 |
| S1c | 0.7022 | 0.7017 | +0.0005 | ±0.0018 | ±0.0031 | +0.21 | 6 of 10 | 1 |
| S2c | 0.6609 | 0.6589 | +0.0020 | ±0.0044 | ±0.0075 | +0.33 | 8 of 10 | 1 |
| wine | 0.5417 | 0.5413 | +0.0004 | ±0.0021 | ±0.0036 | +0.13 | 6 of 10 | 1 |
| breast_cancer | 0.7094 | 0.7061 | +0.0034 | ±0.0006 (excludes 0) | ±0.0011 (excludes 0) | +3.76 | 10 of 10 | 0.0254 |
| thyroid | 0.8925 | 0.8920 | +0.0005 | ±0.0013 | ±0.0022 | +0.30 | 5 of 10 | 1 |
| shuttle | 0.7063 | 0.7059 | +0.0004 | ±0.0011 | ±0.0019 | +0.24 | 6 of 10 | 1 |
| magic | 0.2677 | 0.2668 | +0.0009 | ±0.0046 | ±0.0078 | +0.14 | 7 of 10 | 1 |
| letter | -0.0150 | -0.0165 | +0.0015 | ±0.0036 | ±0.0061 | +0.29 | 6 of 10 | 1 |
| landsat | 0.2485 | 0.2129 | +0.0357 | ±0.0069 (excludes 0) | ±0.0117 (excludes 0) | +3.72 | 10 of 10 | 0.0254 |
| page_blocks | 0.2756 | 0.2732 | +0.0024 | ±0.0038 | ±0.0064 | +0.46 | 7 of 10 | 1 |

Milliseconds per explanation, KernelSHAP divided by PACE-SHAP (runs shared the CPU; each pair of games shares its model calls): S1 5.3, S2 6.9, S3 6.8, S1c 6.8, S2c 7.2, wine 8.6, breast_cancer 6.3, thyroid 7.3, shuttle 8.9, magic 8.5, letter 10.0, landsat 4.6, page_blocks 3.5.

## M4 and m4. Absolute AUCs, effect sizes and adjusted intervals (v3)

From `faithfulness_v3/runs.csv` (seeds 30–39). Share of the gap to Random: how much of KernelSHAP's distance from Random PACE-SHAP adds (positive: PACE-SHAP is further from Random). Figure 2 draws the unadjusted 95% intervals while its filled markers use Holm; the Bonferroni interval is the adjusted version (11 datasets).

### Mean deletion margin AUC (lower is more faithful)

| Dataset | PACE-SHAP | PACE-SHAP-20 | KernelSHAP | KernelSHAP-strong | LIME | PACE-flip | Random |
|---|---|---|---|---|---|---|---|
| S1 | -0.1389 | -0.1314 | -0.1326 | -0.1314 | -0.1095 | -0.0892 | 0.2708 |
| S2 | -0.1317 | -0.1285 | -0.1249 | -0.1285 | -0.0951 | -0.0755 | 0.2565 |
| S3 | -0.0552 | -0.0440 | -0.0412 | -0.0440 | 0.0117 | 0.0057 | 0.3179 |
| wine | -0.0761 | -0.0701 | -0.0684 | -0.0702 | -0.0387 | -0.0545 | 0.2356 |
| breast_cancer | 0.1114 | 0.1221 | 0.1743 | 0.1222 | 0.1412 | 0.1366 | 0.3981 |
| thyroid | 0.5486 | 0.5624 | 0.5748 | 0.5624 | 0.5708 | 0.5440 | 0.7392 |
| shuttle | 0.3701 | 0.3654 | 0.3652 | 0.3657 | 0.3812 | 0.3741 | 0.5516 |
| magic | -0.0522 | -0.0377 | -0.0348 | -0.0377 | -0.0115 | 0.0153 | 0.1400 |
| letter | -0.6897 | -0.6842 | -0.6776 | -0.6838 | -0.6561 | -0.6441 | -0.3811 |
| landsat | -0.3287 | -0.3093 | -0.2166 | -0.3031 | -0.2954 | -0.2546 | -0.0497 |
| page_blocks | 0.0501 | 0.0422 | 0.0464 | 0.0426 | 0.0487 | 0.0514 | 0.1282 |

### Mean insertion margin AUC (higher is more faithful)

| Dataset | PACE-SHAP | PACE-SHAP-20 | KernelSHAP | KernelSHAP-strong | LIME | PACE-flip | Random |
|---|---|---|---|---|---|---|---|
| S1 | 0.6455 | 0.6363 | 0.6351 | 0.6363 | 0.6092 | 0.6168 | 0.2957 |
| S2 | 0.6095 | 0.5930 | 0.5917 | 0.5930 | 0.5648 | 0.5611 | 0.2566 |
| S3 | 0.6392 | 0.6266 | 0.6264 | 0.6266 | 0.5688 | 0.5950 | 0.3137 |
| wine | 0.5327 | 0.5261 | 0.5241 | 0.5262 | 0.4920 | 0.4890 | 0.2386 |
| breast_cancer | 0.6993 | 0.6873 | 0.6337 | 0.6870 | 0.6803 | 0.6853 | 0.4142 |
| thyroid | 0.9147 | 0.9019 | 0.8916 | 0.9026 | 0.8737 | 0.9177 | 0.7357 |
| shuttle | 0.7439 | 0.7470 | 0.7468 | 0.7469 | 0.7210 | 0.7504 | 0.5362 |
| magic | 0.2902 | 0.2794 | 0.2721 | 0.2794 | 0.2346 | 0.2342 | 0.0928 |
| letter | -0.0005 | -0.0207 | -0.0379 | -0.0219 | -0.1358 | -0.0724 | -0.4229 |
| landsat | 0.2308 | 0.2057 | 0.0772 | 0.2035 | 0.1257 | 0.1340 | -0.0572 |
| page_blocks | 0.2036 | 0.2120 | 0.2045 | 0.2116 | 0.1965 | 0.2017 | 0.0954 |

### PACE-SHAP vs KernelSHAP, deletion

PACE-SHAP minus KernelSHAP in `margin_auc` (lower is better). **PACE-SHAP is better on 6, level on 5 and worse on 0** (Holm across 11).

| Dataset | PACE-SHAP | KernelSHAP | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE-SHAP | p (Holm) | Share of the gap to Random |
|---|---|---|---|---|---|---|---|---|---|
| S1 | -0.1389 | -0.1326 | -0.0063 | ±0.0074 | ±0.0122 | -0.61 | 8 of 10 | 0.146 | 2% |
| S2 | -0.1317 | -0.1249 | -0.0068 | ±0.0059 (excludes 0) | ±0.0099 | -0.82 | 7 of 10 | 0.137 | 2% |
| S3 | -0.0552 | -0.0412 | -0.0140 | ±0.0082 (excludes 0) | ±0.0135 (excludes 0) | -1.22 | 9 of 10 | 0.0273 | 4% |
| wine | -0.0761 | -0.0684 | -0.0078 | ±0.0045 (excludes 0) | ±0.0075 (excludes 0) | -1.23 | 10 of 10 | 0.0215 | 3% |
| breast_cancer | 0.1114 | 0.1743 | -0.0630 | ±0.0052 (excludes 0) | ±0.0087 (excludes 0) | -8.63 | 10 of 10 | 0.0215 | 28% |
| thyroid | 0.5486 | 0.5748 | -0.0262 | ±0.0108 (excludes 0) | ±0.0179 (excludes 0) | -1.73 | 10 of 10 | 0.0215 | 16% |
| shuttle | 0.3701 | 0.3652 | +0.0048 | ±0.0082 | ±0.0136 | +0.42 | 4 of 10 | 0.551 | -3% |
| magic | -0.0522 | -0.0348 | -0.0174 | ±0.0142 (excludes 0) | ±0.0236 | -0.87 | 8 of 10 | 0.137 | 10% |
| letter | -0.6897 | -0.6776 | -0.0121 | ±0.0066 (excludes 0) | ±0.0110 (excludes 0) | -1.30 | 9 of 10 | 0.0352 | 4% |
| landsat | -0.3287 | -0.2166 | -0.1121 | ±0.0204 (excludes 0) | ±0.0338 (excludes 0) | -3.94 | 10 of 10 | 0.0215 | 67% |
| page_blocks | 0.0501 | 0.0464 | +0.0037 | ±0.0180 | ±0.0299 | +0.15 | 5 of 10 | 0.625 | -5% |

### PACE-SHAP vs KernelSHAP, insertion

PACE-SHAP minus KernelSHAP in `insertion_margin_auc` (higher is better). **PACE-SHAP is better on 7, level on 4 and worse on 0** (Holm across 11).

| Dataset | PACE-SHAP | KernelSHAP | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE-SHAP | p (Holm) | Share of the gap to Random |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 0.6455 | 0.6351 | +0.0104 | ±0.0058 (excludes 0) | ±0.0096 (excludes 0) | +1.29 | 10 of 10 | 0.0215 | 3% |
| S2 | 0.6095 | 0.5917 | +0.0178 | ±0.0118 (excludes 0) | ±0.0196 | +1.07 | 10 of 10 | 0.0215 | 5% |
| S3 | 0.6392 | 0.6264 | +0.0128 | ±0.0080 (excludes 0) | ±0.0132 | +1.15 | 8 of 10 | 0.0781 | 4% |
| wine | 0.5327 | 0.5241 | +0.0086 | ±0.0056 (excludes 0) | ±0.0092 | +1.10 | 9 of 10 | 0.0293 | 3% |
| breast_cancer | 0.6993 | 0.6337 | +0.0656 | ±0.0097 (excludes 0) | ±0.0161 (excludes 0) | +4.82 | 10 of 10 | 0.0215 | 30% |
| thyroid | 0.9147 | 0.8916 | +0.0231 | ±0.0097 (excludes 0) | ±0.0161 (excludes 0) | +1.70 | 10 of 10 | 0.0215 | 15% |
| shuttle | 0.7439 | 0.7468 | -0.0029 | ±0.0095 | ±0.0158 | -0.22 | 5 of 10 | 0.984 | -1% |
| magic | 0.2902 | 0.2721 | +0.0181 | ±0.0133 (excludes 0) | ±0.0221 | +0.97 | 8 of 10 | 0.082 | 10% |
| letter | -0.0005 | -0.0379 | +0.0374 | ±0.0083 (excludes 0) | ±0.0137 (excludes 0) | +3.24 | 10 of 10 | 0.0215 | 10% |
| landsat | 0.2308 | 0.0772 | +0.1537 | ±0.0225 (excludes 0) | ±0.0372 (excludes 0) | +4.89 | 10 of 10 | 0.0215 | 114% |
| page_blocks | 0.2036 | 0.2045 | -0.0010 | ±0.0208 | ±0.0345 | -0.03 | 6 of 10 | 0.984 | -1% |

### PACE-SHAP vs KernelSHAP-strong, deletion

PACE-SHAP minus KernelSHAP-strong in `margin_auc` (lower is better). **PACE-SHAP is better on 3, level on 8 and worse on 0** (Holm across 11).

| Dataset | PACE-SHAP | KernelSHAP-strong | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE-SHAP | p (Holm) | Share of the gap to Random |
|---|---|---|---|---|---|---|---|---|---|
| S1 | -0.1389 | -0.1314 | -0.0075 | ±0.0077 | ±0.0128 | -0.70 | 7 of 10 | 0.26 | 2% |
| S2 | -0.1317 | -0.1285 | -0.0033 | ±0.0048 | ±0.0080 | -0.48 | 7 of 10 | 0.422 | 1% |
| S3 | -0.0552 | -0.0440 | -0.0112 | ±0.0071 (excludes 0) | ±0.0118 | -1.13 | 8 of 10 | 0.0781 | 3% |
| wine | -0.0761 | -0.0702 | -0.0059 | ±0.0050 (excludes 0) | ±0.0083 | -0.85 | 7 of 10 | 0.293 | 2% |
| breast_cancer | 0.1114 | 0.1222 | -0.0109 | ±0.0052 (excludes 0) | ±0.0087 (excludes 0) | -1.49 | 10 of 10 | 0.0215 | 4% |
| thyroid | 0.5486 | 0.5624 | -0.0138 | ±0.0068 (excludes 0) | ±0.0112 (excludes 0) | -1.46 | 9 of 10 | 0.0352 | 8% |
| shuttle | 0.3701 | 0.3657 | +0.0044 | ±0.0074 | ±0.0123 | +0.42 | 3 of 10 | 0.465 | -2% |
| magic | -0.0522 | -0.0377 | -0.0145 | ±0.0145 (excludes 0) | ±0.0240 | -0.72 | 6 of 10 | 0.422 | 8% |
| letter | -0.6897 | -0.6838 | -0.0059 | ±0.0054 (excludes 0) | ±0.0089 | -0.78 | 8 of 10 | 0.293 | 2% |
| landsat | -0.3287 | -0.3031 | -0.0256 | ±0.0136 (excludes 0) | ±0.0226 (excludes 0) | -1.34 | 10 of 10 | 0.0215 | 10% |
| page_blocks | 0.0501 | 0.0426 | +0.0075 | ±0.0173 | ±0.0286 | +0.31 | 4 of 10 | 0.625 | -9% |

### PACE-SHAP vs KernelSHAP-strong, insertion

PACE-SHAP minus KernelSHAP-strong in `insertion_margin_auc` (higher is better). **PACE-SHAP is better on 4, level on 7 and worse on 0** (Holm across 11).

| Dataset | PACE-SHAP | KernelSHAP-strong | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE-SHAP | p (Holm) | Share of the gap to Random |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 0.6455 | 0.6363 | +0.0092 | ±0.0047 (excludes 0) | ±0.0078 (excludes 0) | +1.40 | 10 of 10 | 0.0215 | 3% |
| S2 | 0.6095 | 0.5930 | +0.0164 | ±0.0116 (excludes 0) | ±0.0193 | +1.01 | 8 of 10 | 0.0684 | 5% |
| S3 | 0.6392 | 0.6266 | +0.0126 | ±0.0064 (excludes 0) | ±0.0107 (excludes 0) | +1.39 | 10 of 10 | 0.0215 | 4% |
| wine | 0.5327 | 0.5262 | +0.0065 | ±0.0056 (excludes 0) | ±0.0093 | +0.83 | 8 of 10 | 0.0684 | 2% |
| breast_cancer | 0.6993 | 0.6870 | +0.0123 | ±0.0055 (excludes 0) | ±0.0091 (excludes 0) | +1.60 | 10 of 10 | 0.0215 | 5% |
| thyroid | 0.9147 | 0.9026 | +0.0122 | ±0.0093 (excludes 0) | ±0.0154 | +0.94 | 8 of 10 | 0.109 | 7% |
| shuttle | 0.7439 | 0.7469 | -0.0029 | ±0.0094 | ±0.0155 | -0.22 | 4 of 10 | 0.984 | -1% |
| magic | 0.2902 | 0.2794 | +0.0107 | ±0.0132 | ±0.0219 | +0.58 | 6 of 10 | 0.48 | 6% |
| letter | -0.0005 | -0.0219 | +0.0214 | ±0.0078 (excludes 0) | ±0.0130 (excludes 0) | +1.96 | 10 of 10 | 0.0215 | 5% |
| landsat | 0.2308 | 0.2035 | +0.0274 | ±0.0209 (excludes 0) | ±0.0347 | +0.94 | 9 of 10 | 0.0684 | 10% |
| page_blocks | 0.2036 | 0.2116 | -0.0080 | ±0.0228 | ±0.0378 | -0.25 | 3 of 10 | 0.984 | -7% |

## M2. Stability when both methods re-draw 20 rows (v3)

v3's re-run re-drew the 20 rows of KernelSHAP and of PACE-SHAP-20, but kept PACE-SHAP's 200. Stability: Jaccard overlap of the top 3 features between the two runs.

| Dataset | PACE-SHAP | PACE-SHAP-20 | KernelSHAP | LIME | PACE-flip |
|---|---|---|---|---|---|
| S1 | 1.000 | 0.736 | 0.728 | 0.897 | 1.000 |
| S2 | 1.000 | 0.670 | 0.668 | 0.879 | 1.000 |
| S3 | 1.000 | 0.686 | 0.650 | 0.855 | 1.000 |
| wine | 0.975 | 0.709 | 0.708 | 0.885 | 1.000 |
| breast_cancer | 0.967 | 0.545 | 0.375 | 0.717 | 1.000 |
| thyroid | 0.990 | 0.552 | 0.546 | 0.765 | 1.000 |
| shuttle | 1.000 | 0.588 | 0.593 | 0.900 | 1.000 |
| magic | 1.000 | 0.498 | 0.466 | 0.689 | 1.000 |
| letter | 0.987 | 0.630 | 0.608 | 0.846 | 1.000 |
| landsat | 0.936 | 0.516 | 0.148 | 0.524 | 1.000 |
| page_blocks | 1.000 | 0.481 | 0.467 | 0.792 | 1.000 |

PACE-SHAP-20 minus KernelSHAP in `stability_top3` (higher is better). **PACE-SHAP-20 is better on 5, level on 6 and worse on 0** (Holm across 11).

| Dataset | PACE-SHAP-20 | KernelSHAP | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE-SHAP-20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| S1 | 0.7360 | 0.7284 | +0.0076 | ±0.0211 | ±0.0351 | +0.26 | 5 of 10 | 1 |
| S2 | 0.6698 | 0.6678 | +0.0020 | ±0.0197 | ±0.0326 | +0.07 | 4 of 10 | 1 |
| S3 | 0.6860 | 0.6496 | +0.0364 | ±0.0198 (excludes 0) | ±0.0329 (excludes 0) | +1.31 | 9 of 10 | 0.0273 |
| wine | 0.7089 | 0.7083 | +0.0006 | ±0.0168 | ±0.0278 | +0.02 | 4 of 10 | 1 |
| breast_cancer | 0.5454 | 0.3754 | +0.1700 | ±0.0575 (excludes 0) | ±0.0953 (excludes 0) | +2.12 | 10 of 10 | 0.0215 |
| thyroid | 0.5522 | 0.5460 | +0.0062 | ±0.0067 | ±0.0111 | +0.66 | 5 of 10 | 0.469 |
| shuttle | 0.5882 | 0.5930 | -0.0048 | ±0.0114 | ±0.0189 | -0.30 | 3 of 10 | 1 |
| magic | 0.4978 | 0.4658 | +0.0320 | ±0.0178 (excludes 0) | ±0.0295 (excludes 0) | +1.29 | 10 of 10 | 0.0215 |
| letter | 0.6300 | 0.6080 | +0.0220 | ±0.0121 (excludes 0) | ±0.0200 (excludes 0) | +1.30 | 10 of 10 | 0.0215 |
| landsat | 0.5162 | 0.1484 | +0.3678 | ±0.0652 (excludes 0) | ±0.1080 (excludes 0) | +4.04 | 10 of 10 | 0.0215 |
| page_blocks | 0.4814 | 0.4672 | +0.0142 | ±0.0222 | ±0.0368 | +0.46 | 5 of 10 | 0.977 |

## M3. Tied explained points (v3's models, seeds 30–39)

A point is tied when its margin is 0 (at most 1e-12): its class is then decided by the nearest centroid, which the explained margin ignores.

| Dataset | Points | Tied | Margin below 0.01 |
|---|---|---|---|
| S1 | 500 | 0% | 8% |
| S2 | 500 | 0% | 9% |
| S3 | 500 | 0% | 7% |
| wine | 360 | 0% | 0% |
| breast_cancer | 500 | 1% | 12% |
| thyroid | 500 | 5% | 79% |
| shuttle | 500 | 11% | 94% |
| magic | 500 | 75% | 89% |
| letter | 500 | 55% | 55% |
| landsat | 500 | 25% | 49% |
| page_blocks | 500 | 79% | 90% |

## M1. PACE-SHAP's sampled mode (from `faithfulness_v3/settings.md`, seed 99)

Error against a reference with 128 orders per row (not exact values; Test 1 adds that).

| Dataset | d | Orders per row | Relative L1 error | Top-3 overlap |
|---|---|---|---|---|
| wine | 13 | 2 | 0.035 | 0.97 |
| wine | 13 | 4 | 0.025 | 0.95 |
| wine | 13 | 8 | 0.020 | 1.00 |
| wine | 13 | 16 | 0.014 | 0.97 |
| breast_cancer | 30 | 2 | 0.044 | 0.93 |
| breast_cancer | 30 | 4 | 0.030 | 0.93 |
| breast_cancer | 30 | 8 | 0.021 | 0.95 |
| breast_cancer | 30 | 16 | 0.017 | 0.97 |
| thyroid | 21 | 2 | 0.052 | 0.82 |
| thyroid | 21 | 4 | 0.039 | 0.80 |
| thyroid | 21 | 8 | 0.029 | 0.91 |
| thyroid | 21 | 16 | 0.022 | 0.85 |
| letter | 16 | 2 | 0.052 | 1.00 |
| letter | 16 | 4 | 0.043 | 0.97 |
| letter | 16 | 8 | 0.028 | 0.95 |
| letter | 16 | 16 | 0.021 | 0.97 |
| landsat | 36 | 2 | 0.058 | 0.88 |
| landsat | 36 | 4 | 0.043 | 0.82 |
| landsat | 36 | 8 | 0.028 | 0.95 |
| landsat | 36 | 16 | 0.020 | 0.93 |

## M6. Counterfactuals: PACE vs DiCE, tested over seeds

From B1 (`extensions_v1/counterfactuals/runs.csv`, seeds 40–49), on the points where both found a valid counterfactual. The first counterfactual run tested over points pooled from 3 models; here each seed is one observation.

### Size of the change (L1)

On the 8 real datasets: PACE better on 5, level on 3, worse on 0.

PACE minus DiCE in `l1` (lower is better). **PACE is better on 5, level on 6 and worse on 0** (Holm across 11).

| Dataset | PACE | DiCE | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| S1 | 0.3858 | 0.4096 | -0.0238 | ±0.0352 | ±0.0583 | -0.48 | 7 of 10 | 0.387 |
| S2 | 0.3629 | 0.3909 | -0.0280 | ±0.0250 (excludes 0) | ±0.0415 | -0.80 | 9 of 10 | 0.336 |
| S3 | 0.3698 | 0.3950 | -0.0252 | ±0.0298 | ±0.0493 | -0.61 | 7 of 10 | 0.336 |
| wine | 0.7012 | 0.8042 | -0.1030 | ±0.0323 (excludes 0) | ±0.0536 (excludes 0) | -2.28 | 10 of 10 | 0.0215 |
| breast_cancer | 1.4265 | 1.6599 | -0.2334 | ±0.1426 (excludes 0) | ±0.2364 | -1.17 | 8 of 10 | 0.0684 |
| thyroid | 0.1564 | 0.2924 | -0.1360 | ±0.0304 (excludes 0) | ±0.0503 (excludes 0) | -3.21 | 10 of 10 | 0.0215 |
| shuttle | 0.1170 | 0.2478 | -0.1308 | ±0.0275 (excludes 0) | ±0.0456 (excludes 0) | -3.40 | 10 of 10 | 0.0215 |
| magic | 0.2503 | 0.3269 | -0.0766 | ±0.0327 (excludes 0) | ±0.0543 (excludes 0) | -1.67 | 9 of 10 | 0.0273 |
| letter | 0.2136 | 0.3217 | -0.1082 | ±0.0236 (excludes 0) | ±0.0391 (excludes 0) | -3.28 | 10 of 10 | 0.0215 |
| landsat | 1.0159 | 1.1003 | -0.0844 | ±0.1344 | ±0.2228 | -0.45 | 6 of 10 | 0.387 |
| page_blocks | 0.2386 | 0.3484 | -0.1098 | ±0.0849 (excludes 0) | ±0.1408 | -0.93 | 9 of 10 | 0.0586 |

### Features changed

On the 8 real datasets: PACE better on 0, level on 0, worse on 8.

PACE minus DiCE in `n_changed` (lower is better). **PACE is better on 0, level on 1 and worse on 10** (Holm across 11).

| Dataset | PACE | DiCE | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| S1 | 1.5767 | 1.3167 | +0.2600 | ±0.1194 (excludes 0) | ±0.1980 (excludes 0) | +1.56 | 1 of 10 | 0.0215 |
| S2 | 1.5567 | 1.3167 | +0.2400 | ±0.1123 (excludes 0) | ±0.1862 (excludes 0) | +1.53 | 0 of 10 | 0.0215 |
| S3 | 1.4433 | 1.3600 | +0.0833 | ±0.1238 | ±0.2053 | +0.48 | 4 of 10 | 0.238 |
| wine | 3.4400 | 2.3467 | +1.0933 | ±0.2702 (excludes 0) | ±0.4481 (excludes 0) | +2.89 | 0 of 10 | 0.0215 |
| breast_cancer | 10.9567 | 4.5067 | +6.4500 | ±0.6840 (excludes 0) | ±1.1342 (excludes 0) | +6.75 | 0 of 10 | 0.0215 |
| thyroid | 1.9036 | 1.6048 | +0.2988 | ±0.1193 (excludes 0) | ±0.1979 (excludes 0) | +1.79 | 1 of 10 | 0.0215 |
| shuttle | 3.2933 | 1.3667 | +1.9267 | ±0.2087 (excludes 0) | ±0.3461 (excludes 0) | +6.60 | 0 of 10 | 0.0215 |
| magic | 2.5600 | 1.4100 | +1.1500 | ±0.2613 (excludes 0) | ±0.4333 (excludes 0) | +3.15 | 0 of 10 | 0.0215 |
| letter | 2.6555 | 1.4524 | +1.2031 | ±0.3341 (excludes 0) | ±0.5540 (excludes 0) | +2.58 | 0 of 10 | 0.0215 |
| landsat | 10.6389 | 3.3151 | +7.3238 | ±0.6338 (excludes 0) | ±1.0511 (excludes 0) | +8.27 | 0 of 10 | 0.0215 |
| page_blocks | 2.1817 | 1.3740 | +0.8076 | ±0.1899 (excludes 0) | ±0.3149 (excludes 0) | +3.04 | 0 of 10 | 0.0215 |

### Distance to real data (plausibility)

On the 8 real datasets: PACE better on 8, level on 0, worse on 0.

PACE minus DiCE in `plausibility` (lower is better). **PACE is better on 11, level on 0 and worse on 0** (Holm across 11).

| Dataset | PACE | DiCE | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| S1 | 1.1624 | 1.2258 | -0.0634 | ±0.0342 (excludes 0) | ±0.0568 (excludes 0) | -1.33 | 10 of 10 | 0.0215 |
| S2 | 1.1141 | 1.1943 | -0.0802 | ±0.0332 (excludes 0) | ±0.0551 (excludes 0) | -1.73 | 10 of 10 | 0.0215 |
| S3 | 1.1381 | 1.2362 | -0.0981 | ±0.0233 (excludes 0) | ±0.0386 (excludes 0) | -3.01 | 10 of 10 | 0.0215 |
| wine | 1.3173 | 1.6714 | -0.3541 | ±0.0507 (excludes 0) | ±0.0840 (excludes 0) | -5.00 | 10 of 10 | 0.0215 |
| breast_cancer | 1.6826 | 3.0137 | -1.3311 | ±0.1162 (excludes 0) | ±0.1927 (excludes 0) | -8.20 | 10 of 10 | 0.0215 |
| thyroid | 0.2459 | 0.5133 | -0.2674 | ±0.0627 (excludes 0) | ±0.1040 (excludes 0) | -3.05 | 10 of 10 | 0.0215 |
| shuttle | 0.1538 | 0.2850 | -0.1312 | ±0.0193 (excludes 0) | ±0.0319 (excludes 0) | -4.87 | 10 of 10 | 0.0215 |
| magic | 0.2554 | 0.4569 | -0.2015 | ±0.0175 (excludes 0) | ±0.0291 (excludes 0) | -8.23 | 10 of 10 | 0.0215 |
| letter | 0.7248 | 0.9203 | -0.1955 | ±0.0222 (excludes 0) | ±0.0369 (excludes 0) | -6.29 | 10 of 10 | 0.0215 |
| landsat | 1.6908 | 2.5293 | -0.8385 | ±0.2015 (excludes 0) | ±0.3341 (excludes 0) | -2.98 | 10 of 10 | 0.0215 |
| page_blocks | 0.2358 | 0.5118 | -0.2760 | ±0.0613 (excludes 0) | ±0.1017 (excludes 0) | -3.22 | 10 of 10 | 0.0215 |

## m6. Do the geometry signals add anything? (RQ2 combination run, seeds 20–29)

LR: all minus LR: kNN + TS + order ens. in `auprc` (higher is better). **LR: all is better on 1, level on 7 and worse on 0** (Holm across 8).

| Dataset | LR: all | LR: kNN + TS + order ens. | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by LR: all | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| Synthetic family | 0.7708 | 0.7737 | -0.0029 | ±0.0032 | ±0.0050 | -0.65 | 4 of 10 | 0.801 |
| digits | 0.8415 | 0.8530 | -0.0115 | ±0.0200 | ±0.0313 | -0.41 | 5 of 10 | 0.93 |
| thyroid | 0.5020 | 0.4583 | +0.0436 | ±0.0676 | ±0.1060 | +0.46 | 6 of 10 | 0.93 |
| skin | 0.9977 | 0.9984 | -0.0007 | ±0.0009 | ±0.0014 | -0.58 | 4 of 10 | 0.93 |
| magic | 0.7493 | 0.7488 | +0.0005 | ±0.0017 | ±0.0026 | +0.20 | 5 of 10 | 0.93 |
| letter | 0.9841 | 0.9826 | +0.0015 | ±0.0005 (excludes 0) | ±0.0007 (excludes 0) | +2.28 | 10 of 10 | 0.0156 |
| landsat | 0.7830 | 0.7869 | -0.0040 | ±0.0050 | ±0.0078 | -0.57 | 2 of 10 | 0.504 |
| page_blocks | 0.8576 | 0.8699 | -0.0123 | ±0.0108 (excludes 0) | ±0.0169 | -0.81 | 1 of 10 | 0.0957 |

## m7. Conditional vs interventional PACE-SHAP on the conditional test (B3)

PACE-SHAP-cond minus PACE-SHAP in `cond_margin_auc` (lower is better). **PACE-SHAP-cond is better on 1, level on 2 and worse on 10** (Holm across 13).

| Dataset | PACE-SHAP-cond | PACE-SHAP | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE-SHAP-cond | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| S1 | -0.1311 | -0.1838 | +0.0528 | ±0.0124 (excludes 0) | ±0.0211 (excludes 0) | +3.05 | 0 of 10 | 0.0254 |
| S2 | -0.1096 | -0.1724 | +0.0628 | ±0.0143 (excludes 0) | ±0.0245 (excludes 0) | +3.13 | 0 of 10 | 0.0254 |
| S3 | -0.0390 | -0.0892 | +0.0502 | ±0.0083 (excludes 0) | ±0.0141 (excludes 0) | +4.35 | 0 of 10 | 0.0254 |
| S1c | -0.0515 | -0.0941 | +0.0426 | ±0.0112 (excludes 0) | ±0.0192 (excludes 0) | +2.71 | 0 of 10 | 0.0254 |
| S2c | -0.1050 | -0.1567 | +0.0516 | ±0.0078 (excludes 0) | ±0.0133 (excludes 0) | +4.73 | 0 of 10 | 0.0254 |
| wine | 0.1483 | 0.0798 | +0.0685 | ±0.0146 (excludes 0) | ±0.0249 (excludes 0) | +3.36 | 0 of 10 | 0.0254 |
| breast_cancer | 0.3601 | 0.2486 | +0.1115 | ±0.0161 (excludes 0) | ±0.0275 (excludes 0) | +4.94 | 0 of 10 | 0.0254 |
| thyroid | 0.5564 | 0.5279 | +0.0286 | ±0.0160 (excludes 0) | ±0.0272 (excludes 0) | +1.28 | 2 of 10 | 0.0293 |
| shuttle | 0.4522 | 0.4817 | -0.0295 | ±0.0131 (excludes 0) | ±0.0223 (excludes 0) | -1.61 | 9 of 10 | 0.0254 |
| magic | 0.0472 | 0.0500 | -0.0028 | ±0.0121 | ±0.0206 | -0.17 | 6 of 10 | 0.863 |
| letter | -0.5246 | -0.6003 | +0.0757 | ±0.0120 (excludes 0) | ±0.0205 (excludes 0) | +4.50 | 0 of 10 | 0.0254 |
| landsat | 0.2654 | 0.1597 | +0.1058 | ±0.0257 (excludes 0) | ±0.0438 (excludes 0) | +2.94 | 0 of 10 | 0.0254 |
| page_blocks | 0.1136 | 0.1097 | +0.0039 | ±0.0137 | ±0.0233 | +0.20 | 5 of 10 | 0.863 |

PACE-SHAP-cond minus PACE-SHAP in `cond_insertion_margin_auc` (higher is better). **PACE-SHAP-cond is better on 0, level on 4 and worse on 9** (Holm across 13).

| Dataset | PACE-SHAP-cond | PACE-SHAP | Difference | 95% CI | Bonferroni CI | d_z | Seeds won by PACE-SHAP-cond | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| S1 | 0.7382 | 0.7777 | -0.0395 | ±0.0063 (excludes 0) | ±0.0107 (excludes 0) | -4.50 | 0 of 10 | 0.0254 |
| S2 | 0.6599 | 0.7135 | -0.0536 | ±0.0109 (excludes 0) | ±0.0186 (excludes 0) | -3.51 | 0 of 10 | 0.0254 |
| S3 | 0.7114 | 0.7517 | -0.0403 | ±0.0072 (excludes 0) | ±0.0123 (excludes 0) | -4.00 | 0 of 10 | 0.0254 |
| S1c | 0.7871 | 0.8120 | -0.0249 | ±0.0052 (excludes 0) | ±0.0089 (excludes 0) | -3.40 | 0 of 10 | 0.0254 |
| S2c | 0.7062 | 0.7431 | -0.0368 | ±0.0052 (excludes 0) | ±0.0089 (excludes 0) | -5.06 | 0 of 10 | 0.0254 |
| wine | 0.8568 | 0.8907 | -0.0338 | ±0.0080 (excludes 0) | ±0.0137 (excludes 0) | -3.02 | 0 of 10 | 0.0254 |
| breast_cancer | 0.9152 | 0.9332 | -0.0180 | ±0.0063 (excludes 0) | ±0.0107 (excludes 0) | -2.04 | 0 of 10 | 0.0254 |
| thyroid | 0.8754 | 0.9073 | -0.0319 | ±0.0094 (excludes 0) | ±0.0160 (excludes 0) | -2.43 | 0 of 10 | 0.0254 |
| shuttle | 0.7708 | 0.7727 | -0.0020 | ±0.0108 | ±0.0185 | -0.13 | 4 of 10 | 1 |
| magic | 0.1745 | 0.1871 | -0.0126 | ±0.0110 (excludes 0) | ±0.0188 | -0.82 | 1 of 10 | 0.148 |
| letter | -0.0569 | 0.0789 | -0.1358 | ±0.0192 (excludes 0) | ±0.0328 (excludes 0) | -5.06 | 0 of 10 | 0.0254 |
| landsat | 0.5984 | 0.5972 | +0.0011 | ±0.0149 | ±0.0254 | +0.05 | 5 of 10 | 1 |
| page_blocks | 0.2531 | 0.2529 | +0.0002 | ±0.0120 | ±0.0205 | +0.01 | 5 of 10 | 1 |

## m8. Figure 4: the lead of the (x0, x1) interaction over the next pair (B4)

| Dataset | Lead as a ratio | Lead as a difference |
|---|---|---|
| S1 | 1.16–2.01 | 0.0005–0.0030 |
| S2 | 1.29–1.53 | 0.0009–0.0016 |
| S3 | 1.58–2.80 | 0.0024–0.0062 |

## m13. Page Blocks' share of errors in the two RQ2 runs

| Run | Seeds | Share of errors per seed | Mean |
|---|---|---|---|
| Confirmatory run | 10–19 | 10: 0.198, 11: 0.114, 12: 0.120, 13: 0.113, 14: 0.237, 15: 0.129, 16: 0.145, 17: 0.153, 18: 0.089, 19: 0.109 | 0.141 |
| Combination run | 20–29 | 20: 0.084, 21: 0.117, 22: 0.084, 23: 0.121, 24: 0.068, 25: 0.119, 26: 0.102, 27: 0.122, 28: 0.105, 29: 0.103 | 0.103 |
