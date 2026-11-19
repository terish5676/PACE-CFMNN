# Peer review: re-analysis of existing results

Written by `scripts/peer_review_reanalysis.py`. Post hoc: nothing here was pre-registered.

Each comparison is paired over seeds. Intervals are 95% t-intervals of the mean difference, not adjusted for the number of datasets. Verdicts use a two-sided Wilcoxon test with Holm correction across the datasets, as in the paper. With 10 seeds the smallest possible two-sided p is 0.002, so after Holm across 11 datasets a verdict needs at least 9 of 10 seeds to agree, and across 13 all 10.

Equivalence margins were fixed in the script before the numbers were computed: for faithfulness, 10% of KernelSHAP's lead over the random ranking on that dataset; for accuracy and macro-F1, 1 point. "Equivalent" means both one-sided tests against the margin reject after Holm correction.

## 1. RQ1 with effect sizes and equivalence (C1, C2)

The second explanation run (seeds 30–39, 11 datasets) and the like-for-like comparison from the conditional run (seeds 40–49; 13 datasets: the 11, plus S1c and S2c, which are S1 and S2 with feature 3 made a noisy copy of feature 0).

**Usual setting, deletion** (PACE-SHAP − KernelSHAP; lower is better)

**PACE-SHAP is better on 6, not significantly different on 5 and worse on 0** (Holm across 11)
Equivalent within the margin on 6 of 11.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| S1 | -0.0063 | -0.0137 to +0.0011 | -0.61 | 8 of 10 | 0.146 | not significant | 0.0403 | 0.011 (equivalent) |
| S2 | -0.0068 | -0.0127 to -0.0008 | -0.82 | 7 of 10 | 0.137 | not significant | 0.0381 | 0.011 (equivalent) |
| S3 | -0.0140 | -0.0221 to -0.0058 | -1.22 | 9 of 10 | 0.027 | better | 0.0359 | 0.014 (equivalent) |
| wine | -0.0078 | -0.0123 to -0.0032 | -1.23 | 10 of 10 | 0.021 | better | 0.0304 | 0.011 (equivalent) |
| breast_cancer | -0.0630 | -0.0682 to -0.0578 | -8.63 | 10 of 10 | 0.021 | better | 0.0224 | 1.000 |
| thyroid | -0.0262 | -0.0370 to -0.0154 | -1.73 | 10 of 10 | 0.021 | better | 0.0164 | 1.000 |
| shuttle | +0.0048 | -0.0034 to +0.0130 | +0.42 | 4 of 10 | 0.551 | not significant | 0.0186 | 0.029 (equivalent) |
| magic | -0.0174 | -0.0316 to -0.0031 | -0.87 | 8 of 10 | 0.137 | not significant | 0.0175 | 1.000 |
| letter | -0.0121 | -0.0187 to -0.0054 | -1.30 | 9 of 10 | 0.035 | better | 0.0297 | 0.011 (equivalent) |
| landsat | -0.1121 | -0.1325 to -0.0917 | -3.94 | 10 of 10 | 0.021 | better | 0.0167 | 1.000 |
| page_blocks | +0.0037 | -0.0143 to +0.0218 | +0.15 | 5 of 10 | 0.625 | not significant | 0.0082 | 0.483 |

**Stronger KernelSHAP, deletion** (PACE-SHAP − KernelSHAP-strong; lower is better)

**PACE-SHAP is better on 3, not significantly different on 8 and worse on 0** (Holm across 11)
Equivalent within the margin on 7 of 11.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| S1 | -0.0075 | -0.0152 to +0.0002 | -0.70 | 7 of 10 | 0.260 | not significant | 0.0403 | 0.011 (equivalent) |
| S2 | -0.0033 | -0.0081 to +0.0016 | -0.48 | 7 of 10 | 0.422 | not significant | 0.0381 | 0.011 (equivalent) |
| S3 | -0.0112 | -0.0183 to -0.0041 | -1.13 | 8 of 10 | 0.078 | not significant | 0.0359 | 0.011 (equivalent) |
| wine | -0.0059 | -0.0110 to -0.0009 | -0.85 | 7 of 10 | 0.293 | not significant | 0.0304 | 0.011 (equivalent) |
| breast_cancer | -0.0109 | -0.0161 to -0.0057 | -1.49 | 10 of 10 | 0.021 | better | 0.0224 | 0.012 (equivalent) |
| thyroid | -0.0138 | -0.0206 to -0.0070 | -1.46 | 9 of 10 | 0.035 | better | 0.0164 | 0.863 |
| shuttle | +0.0044 | -0.0030 to +0.0118 | +0.42 | 3 of 10 | 0.465 | not significant | 0.0186 | 0.015 (equivalent) |
| magic | -0.0145 | -0.0290 to -0.0000 | -0.72 | 6 of 10 | 0.422 | not significant | 0.0175 | 0.863 |
| letter | -0.0059 | -0.0113 to -0.0005 | -0.78 | 8 of 10 | 0.293 | not significant | 0.0297 | 0.011 (equivalent) |
| landsat | -0.0256 | -0.0392 to -0.0120 | -1.34 | 10 of 10 | 0.021 | better | 0.0167 | 0.903 |
| page_blocks | +0.0075 | -0.0098 to +0.0248 | +0.31 | 4 of 10 | 0.625 | not significant | 0.0082 | 0.863 |

**Same 200 rows, every feature, deletion** (PACE-SHAP − KernelSHAP; lower is better)

**PACE-SHAP is better on 3, not significantly different on 10 and worse on 0** (Holm across 13)
Equivalent within the margin on 12 of 13.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| S1 | -0.0013 | -0.0035 to +0.0008 | -0.44 | 5 of 10 | 1.000 | not significant | 0.0404 | 0.013 (equivalent) |
| S2 | -0.0024 | -0.0054 to +0.0006 | -0.57 | 8 of 10 | 0.756 | not significant | 0.0387 | 0.013 (equivalent) |
| S3 | -0.0066 | -0.0095 to -0.0036 | -1.60 | 10 of 10 | 0.025 | better | 0.0373 | 0.013 (equivalent) |
| S1c | -0.0009 | -0.0054 to +0.0036 | -0.14 | 4 of 10 | 1.000 | not significant | 0.0405 | 0.013 (equivalent) |
| S2c | -0.0022 | -0.0048 to +0.0005 | -0.59 | 7 of 10 | 0.756 | not significant | 0.0405 | 0.013 (equivalent) |
| wine | +0.0007 | -0.0001 to +0.0015 | +0.60 | 3 of 10 | 1.000 | not significant | 0.0316 | 0.013 (equivalent) |
| breast_cancer | -0.0024 | -0.0034 to -0.0013 | -1.63 | 10 of 10 | 0.025 | better | 0.0280 | 0.013 (equivalent) |
| thyroid | +0.0002 | -0.0011 to +0.0014 | +0.09 | 6 of 10 | 1.000 | not significant | 0.0202 | 0.013 (equivalent) |
| shuttle | -0.0006 | -0.0014 to +0.0003 | -0.47 | 8 of 10 | 1.000 | not significant | 0.0186 | 0.013 (equivalent) |
| magic | +0.0023 | -0.0005 to +0.0051 | +0.60 | 2 of 10 | 0.488 | not significant | 0.0179 | 0.013 (equivalent) |
| letter | -0.0007 | -0.0019 to +0.0005 | -0.43 | 9 of 10 | 0.756 | not significant | 0.0294 | 0.013 (equivalent) |
| landsat | -0.0446 | -0.0535 to -0.0357 | -3.60 | 10 of 10 | 0.025 | better | 0.0240 | 1.000 |
| page_blocks | -0.0004 | -0.0030 to +0.0022 | -0.11 | 5 of 10 | 1.000 | not significant | 0.0119 | 0.013 (equivalent) |

**Usual setting, insertion** (PACE-SHAP − KernelSHAP; higher is better)

**PACE-SHAP is better on 7, not significantly different on 4 and worse on 0** (Holm across 11)
Equivalent within the margin on 4 of 11.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| S1 | +0.0104 | +0.0046 to +0.0162 | +1.29 | 10 of 10 | 0.021 | better | 0.0339 | 0.011 (equivalent) |
| S2 | +0.0178 | +0.0059 to +0.0296 | +1.07 | 10 of 10 | 0.021 | better | 0.0335 | 0.096 |
| S3 | +0.0128 | +0.0048 to +0.0208 | +1.15 | 8 of 10 | 0.078 | not significant | 0.0313 | 0.018 (equivalent) |
| wine | +0.0086 | +0.0030 to +0.0141 | +1.10 | 9 of 10 | 0.029 | better | 0.0286 | 0.011 (equivalent) |
| breast_cancer | +0.0656 | +0.0559 to +0.0753 | +4.82 | 10 of 10 | 0.021 | better | 0.0220 | 1.000 |
| thyroid | +0.0231 | +0.0134 to +0.0329 | +1.70 | 10 of 10 | 0.021 | better | 0.0156 | 1.000 |
| shuttle | -0.0029 | -0.0124 to +0.0066 | -0.22 | 5 of 10 | 0.984 | not significant | 0.0211 | 0.018 (equivalent) |
| magic | +0.0181 | +0.0048 to +0.0314 | +0.97 | 8 of 10 | 0.082 | not significant | 0.0179 | 1.000 |
| letter | +0.0374 | +0.0292 to +0.0457 | +3.24 | 10 of 10 | 0.021 | better | 0.0385 | 1.000 |
| landsat | +0.1537 | +0.1312 to +0.1761 | +4.89 | 10 of 10 | 0.021 | better | 0.0134 | 1.000 |
| page_blocks | -0.0010 | -0.0218 to +0.0198 | -0.03 | 6 of 10 | 0.984 | not significant | 0.0109 | 0.967 |

**Stronger KernelSHAP, insertion** (PACE-SHAP − KernelSHAP-strong; higher is better)

**PACE-SHAP is better on 4, not significantly different on 7 and worse on 0** (Holm across 11)
Equivalent within the margin on 7 of 11.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| S1 | +0.0092 | +0.0045 to +0.0139 | +1.40 | 10 of 10 | 0.021 | better | 0.0339 | 0.011 (equivalent) |
| S2 | +0.0164 | +0.0048 to +0.0281 | +1.01 | 8 of 10 | 0.068 | not significant | 0.0335 | 0.034 (equivalent) |
| S3 | +0.0126 | +0.0061 to +0.0190 | +1.39 | 10 of 10 | 0.021 | better | 0.0313 | 0.018 (equivalent) |
| wine | +0.0065 | +0.0009 to +0.0121 | +0.83 | 8 of 10 | 0.068 | not significant | 0.0286 | 0.011 (equivalent) |
| breast_cancer | +0.0123 | +0.0068 to +0.0178 | +1.60 | 10 of 10 | 0.021 | better | 0.0220 | 0.029 (equivalent) |
| thyroid | +0.0122 | +0.0029 to +0.0215 | +0.94 | 8 of 10 | 0.109 | not significant | 0.0156 | 0.647 |
| shuttle | -0.0029 | -0.0123 to +0.0064 | -0.22 | 4 of 10 | 0.984 | not significant | 0.0211 | 0.018 (equivalent) |
| magic | +0.0107 | -0.0025 to +0.0240 | +0.58 | 6 of 10 | 0.480 | not significant | 0.0179 | 0.551 |
| letter | +0.0214 | +0.0136 to +0.0292 | +1.96 | 10 of 10 | 0.021 | better | 0.0385 | 0.018 (equivalent) |
| landsat | +0.0274 | +0.0064 to +0.0483 | +0.94 | 9 of 10 | 0.068 | not significant | 0.0134 | 0.920 |
| page_blocks | -0.0080 | -0.0308 to +0.0147 | -0.25 | 3 of 10 | 0.984 | not significant | 0.0109 | 0.647 |

**Same 200 rows, every feature, insertion** (PACE-SHAP − KernelSHAP; higher is better)

**PACE-SHAP is better on 2, not significantly different on 11 and worse on 0** (Holm across 13)
Equivalent within the margin on 12 of 13.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| S1 | +0.0005 | -0.0018 to +0.0028 | +0.16 | 5 of 10 | 1.000 | not significant | 0.0360 | 0.013 (equivalent) |
| S2 | +0.0036 | +0.0002 to +0.0069 | +0.77 | 8 of 10 | 0.840 | not significant | 0.0340 | 0.013 (equivalent) |
| S3 | +0.0027 | -0.0001 to +0.0055 | +0.68 | 8 of 10 | 0.537 | not significant | 0.0333 | 0.013 (equivalent) |
| S1c | +0.0005 | -0.0013 to +0.0024 | +0.21 | 6 of 10 | 1.000 | not significant | 0.0367 | 0.013 (equivalent) |
| S2c | +0.0020 | -0.0024 to +0.0064 | +0.33 | 8 of 10 | 1.000 | not significant | 0.0383 | 0.013 (equivalent) |
| wine | +0.0004 | -0.0017 to +0.0025 | +0.13 | 6 of 10 | 1.000 | not significant | 0.0305 | 0.013 (equivalent) |
| breast_cancer | +0.0034 | +0.0027 to +0.0040 | +3.76 | 10 of 10 | 0.025 | better | 0.0283 | 0.013 (equivalent) |
| thyroid | +0.0005 | -0.0007 to +0.0018 | +0.30 | 5 of 10 | 1.000 | not significant | 0.0184 | 0.013 (equivalent) |
| shuttle | +0.0004 | -0.0008 to +0.0015 | +0.24 | 6 of 10 | 1.000 | not significant | 0.0217 | 0.013 (equivalent) |
| magic | +0.0009 | -0.0037 to +0.0054 | +0.14 | 7 of 10 | 1.000 | not significant | 0.0184 | 0.013 (equivalent) |
| letter | +0.0015 | -0.0021 to +0.0050 | +0.29 | 6 of 10 | 1.000 | not significant | 0.0396 | 0.013 (equivalent) |
| landsat | +0.0357 | +0.0288 to +0.0425 | +3.72 | 10 of 10 | 0.025 | better | 0.0262 | 0.990 |
| page_blocks | +0.0024 | -0.0014 to +0.0062 | +0.46 | 7 of 10 | 1.000 | not significant | 0.0141 | 0.013 (equivalent) |

## 2. The synthetic family as one unit (C5)

RQ2 averages the synthetic sets because they come from one generator. The same rule in RQ1 gives 9 units.

**Usual setting, deletion, 9 units** (PACE-SHAP − KernelSHAP; lower is better)

**PACE-SHAP is better on 6, not significantly different on 3 and worse on 0** (Holm across 9)
Equivalent within the margin on 4 of 9.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| synthetic | -0.0090 | -0.0145 to -0.0036 | -1.19 | 10 of 10 | 0.018 | better | 0.0381 | 0.009 (equivalent) |
| wine | -0.0078 | -0.0123 to -0.0032 | -1.23 | 10 of 10 | 0.018 | better | 0.0304 | 0.009 (equivalent) |
| breast_cancer | -0.0630 | -0.0682 to -0.0578 | -8.63 | 10 of 10 | 0.018 | better | 0.0224 | 1.000 |
| thyroid | -0.0262 | -0.0370 to -0.0154 | -1.73 | 10 of 10 | 0.018 | better | 0.0164 | 1.000 |
| shuttle | +0.0048 | -0.0034 to +0.0130 | +0.42 | 4 of 10 | 0.551 | not significant | 0.0186 | 0.029 (equivalent) |
| magic | -0.0174 | -0.0316 to -0.0031 | -0.87 | 8 of 10 | 0.082 | not significant | 0.0175 | 1.000 |
| letter | -0.0121 | -0.0187 to -0.0054 | -1.30 | 9 of 10 | 0.023 | better | 0.0297 | 0.009 (equivalent) |
| landsat | -0.1121 | -0.1325 to -0.0917 | -3.94 | 10 of 10 | 0.018 | better | 0.0167 | 1.000 |
| page_blocks | +0.0037 | -0.0143 to +0.0218 | +0.15 | 5 of 10 | 0.625 | not significant | 0.0082 | 0.483 |

**Same rows, deletion, 9 units** (PACE-SHAP − KernelSHAP; lower is better)

**PACE-SHAP is better on 3, not significantly different on 6 and worse on 0** (Holm across 9)
Equivalent within the margin on 8 of 9.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| synthetic | -0.0027 | -0.0040 to -0.0013 | -1.39 | 9 of 10 | 0.041 | better | 0.0395 | 0.009 (equivalent) |
| wine | +0.0007 | -0.0001 to +0.0015 | +0.60 | 3 of 10 | 0.773 | not significant | 0.0316 | 0.009 (equivalent) |
| breast_cancer | -0.0024 | -0.0034 to -0.0013 | -1.63 | 10 of 10 | 0.018 | better | 0.0280 | 0.009 (equivalent) |
| thyroid | +0.0002 | -0.0011 to +0.0014 | +0.09 | 6 of 10 | 1.000 | not significant | 0.0202 | 0.009 (equivalent) |
| shuttle | -0.0006 | -0.0014 to +0.0003 | -0.47 | 8 of 10 | 0.773 | not significant | 0.0186 | 0.009 (equivalent) |
| magic | +0.0023 | -0.0005 to +0.0051 | +0.60 | 2 of 10 | 0.293 | not significant | 0.0179 | 0.009 (equivalent) |
| letter | -0.0007 | -0.0019 to +0.0005 | -0.43 | 9 of 10 | 0.420 | not significant | 0.0294 | 0.009 (equivalent) |
| landsat | -0.0446 | -0.0535 to -0.0357 | -3.60 | 10 of 10 | 0.018 | better | 0.0240 | 1.000 |
| page_blocks | -0.0004 | -0.0030 to +0.0022 | -0.11 | 5 of 10 | 1.000 | not significant | 0.0119 | 0.009 (equivalent) |

**Usual setting, insertion, 9 units** (PACE-SHAP − KernelSHAP; higher is better)

**PACE-SHAP is better on 6, not significantly different on 3 and worse on 0** (Holm across 9)
Equivalent within the margin on 3 of 9.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| synthetic | +0.0137 | +0.0072 to +0.0202 | +1.50 | 10 of 10 | 0.018 | better | 0.0329 | 0.009 (equivalent) |
| wine | +0.0086 | +0.0030 to +0.0141 | +1.10 | 9 of 10 | 0.023 | better | 0.0286 | 0.009 (equivalent) |
| breast_cancer | +0.0656 | +0.0559 to +0.0753 | +4.82 | 10 of 10 | 0.018 | better | 0.0220 | 1.000 |
| thyroid | +0.0231 | +0.0134 to +0.0329 | +1.70 | 10 of 10 | 0.018 | better | 0.0156 | 1.000 |
| shuttle | -0.0029 | -0.0124 to +0.0066 | -0.22 | 5 of 10 | 0.984 | not significant | 0.0211 | 0.014 (equivalent) |
| magic | +0.0181 | +0.0048 to +0.0314 | +0.97 | 8 of 10 | 0.082 | not significant | 0.0179 | 1.000 |
| letter | +0.0374 | +0.0292 to +0.0457 | +3.24 | 10 of 10 | 0.018 | better | 0.0385 | 1.000 |
| landsat | +0.1537 | +0.1312 to +0.1761 | +4.89 | 10 of 10 | 0.018 | better | 0.0134 | 1.000 |
| page_blocks | -0.0010 | -0.0218 to +0.0198 | -0.03 | 6 of 10 | 0.984 | not significant | 0.0109 | 0.967 |

**Same rows, insertion, 9 units** (PACE-SHAP − KernelSHAP; higher is better)

**PACE-SHAP is better on 2, not significantly different on 7 and worse on 0** (Holm across 9)
Equivalent within the margin on 8 of 9.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| synthetic | +0.0019 | +0.0006 to +0.0032 | +1.02 | 9 of 10 | 0.137 | not significant | 0.0357 | 0.009 (equivalent) |
| wine | +0.0004 | -0.0017 to +0.0025 | +0.13 | 6 of 10 | 1.000 | not significant | 0.0305 | 0.009 (equivalent) |
| breast_cancer | +0.0034 | +0.0027 to +0.0040 | +3.76 | 10 of 10 | 0.018 | better | 0.0283 | 0.009 (equivalent) |
| thyroid | +0.0005 | -0.0007 to +0.0018 | +0.30 | 5 of 10 | 1.000 | not significant | 0.0184 | 0.009 (equivalent) |
| shuttle | +0.0004 | -0.0008 to +0.0015 | +0.24 | 6 of 10 | 1.000 | not significant | 0.0217 | 0.009 (equivalent) |
| magic | +0.0009 | -0.0037 to +0.0054 | +0.14 | 7 of 10 | 1.000 | not significant | 0.0184 | 0.009 (equivalent) |
| letter | +0.0015 | -0.0021 to +0.0050 | +0.29 | 6 of 10 | 1.000 | not significant | 0.0396 | 0.009 (equivalent) |
| landsat | +0.0357 | +0.0288 to +0.0425 | +3.72 | 10 of 10 | 0.018 | better | 0.0262 | 0.990 |
| page_blocks | +0.0024 | -0.0014 to +0.0062 | +0.46 | 7 of 10 | 0.961 | not significant | 0.0141 | 0.009 (equivalent) |

## 3. Tests across datasets (C3)

One value per unit (the mean difference over seeds), signed-rank test across the 9 units. For deletion a negative difference favours PACE-SHAP; insertion is given with its sign flipped so that negative again favours PACE-SHAP.

- PACE-SHAP vs KernelSHAP, v3 deletion: 7 of 9 units negative, median difference -0.0121, Wilcoxon p = 0.0195
- PACE-SHAP vs KernelSHAP, same rows deletion: 6 of 9 units negative, median difference -0.0006, Wilcoxon p = 0.2031
- PACE-SHAP vs KernelSHAP, v3 insertion: 7 of 9 units negative, median difference -0.0181, Wilcoxon p = 0.0195
- PACE-SHAP vs KernelSHAP, same rows insertion: 9 of 9 units negative, median difference -0.0015, Wilcoxon p = 0.0039

## 4. Repair with effect sizes and equivalence (C1, C2)

The second repair run (seeds 50–59), test macro-F1 in points.

**Self-repair vs the starting model** (SR − B0; higher is better)

**SR is better on 6, not significantly different on 0 and worse on 0** (Holm across 6)
Equivalent within the margin on 0 of 6.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| thyroid | +6.02 | +2.56 to +9.48 | +1.25 | 9 of 10 | 0.012 | better | 1.00 | 1.000 |
| landsat | +1.48 | +0.20 to +2.75 | +0.83 | 7 of 10 | 0.049 | better | 1.00 | 1.000 |
| page_blocks | +2.58 | +0.81 to +4.35 | +1.04 | 8 of 10 | 0.020 | better | 1.00 | 1.000 |
| magic | +3.27 | +2.17 to +4.37 | +2.13 | 10 of 10 | 0.012 | better | 1.00 | 1.000 |
| letter | +2.25 | +1.78 to +2.72 | +3.45 | 10 of 10 | 0.012 | better | 1.00 | 1.000 |
| skin | +5.66 | +3.71 to +7.60 | +2.08 | 10 of 10 | 0.012 | better | 1.00 | 1.000 |

**SR then AL vs self-repair** (SR+AL − SR; higher is better)

**SR+AL is better on 3, not significantly different on 3 and worse on 0** (Holm across 6)
Equivalent within the margin on 0 of 6.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| thyroid | +0.57 | -0.54 to +1.69 | +0.37 | 5 of 10 | 0.387 | not significant | 1.00 | 0.750 |
| landsat | +1.08 | +0.42 to +1.75 | +1.16 | 8 of 10 | 0.049 | better | 1.00 | 1.000 |
| page_blocks | +2.28 | -1.46 to +6.02 | +0.44 | 7 of 10 | 0.387 | not significant | 1.00 | 1.000 |
| magic | +2.03 | +1.53 to +2.52 | +2.93 | 10 of 10 | 0.012 | better | 1.00 | 1.000 |
| letter | +0.73 | +0.34 to +1.13 | +1.33 | 9 of 10 | 0.049 | better | 1.00 | 0.688 |
| skin | +0.55 | +0.05 to +1.06 | +0.79 | 7 of 10 | 0.193 | not significant | 1.00 | 0.146 |

**Self-repair vs active learning** (SR − AL; higher is better)

**SR is better on 0, not significantly different on 5 and worse on 1** (Holm across 6)
Equivalent within the margin on 0 of 6.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| thyroid | -0.36 | -1.66 to +0.95 | -0.20 | 4 of 10 | 1.000 | not significant | 1.00 | 0.647 |
| landsat | -0.47 | -1.43 to +0.49 | -0.35 | 4 of 10 | 1.000 | not significant | 1.00 | 0.551 |
| page_blocks | -2.30 | -6.95 to +2.34 | -0.35 | 3 of 10 | 1.000 | not significant | 1.00 | 1.000 |
| magic | -1.80 | -2.57 to -1.02 | -1.65 | 0 of 10 | 0.012 | worse | 1.00 | 1.000 |
| letter | +0.33 | -0.22 to +0.87 | +0.43 | 6 of 10 | 1.000 | not significant | 1.00 | 0.082 |
| skin | -0.47 | -1.02 to +0.07 | -0.62 | 4 of 10 | 0.801 | not significant | 1.00 | 0.122 |

**Repair at θ = 0.3 vs θ chosen by validation, no repair (seeds 70–89)** (SR-0.3 − B0-val; higher is better)

**SR-0.3 is better on 1, not significantly different on 5 and worse on 7** (Holm across 13)
Equivalent within the margin on 2 of 13.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| S1 | -1.51 | -2.14 to -0.89 | -1.14 | 3 of 20 | 0.003 | worse | 1.00 | 1.000 |
| S2 | -0.76 | -1.67 to +0.15 | -0.39 | 6 of 20 | 0.413 | not significant | 1.00 | 1.000 |
| S3 | -0.26 | -1.14 to +0.61 | -0.14 | 10 of 20 | 1.000 | not significant | 1.00 | 0.337 |
| wine | +0.14 | -0.85 to +1.13 | +0.07 | 5 of 20 | 1.000 | not significant | 1.00 | 0.999 |
| breast_cancer | -0.26 | -1.21 to +0.68 | -0.13 | 7 of 20 | 1.000 | not significant | 1.00 | 0.449 |
| digits | +0.18 | -0.43 to +0.79 | +0.14 | 7 of 20 | 1.000 | not significant | 1.00 | 0.072 |
| thyroid | +1.07 | +0.78 to +1.35 | +1.76 | 19 of 20 | 0.001 | better | 1.00 | 1.000 |
| shuttle | -0.09 | -0.12 to -0.07 | -1.50 | 0 of 20 | 0.001 | worse | 1.00 | 0.001 (equivalent) |
| skin | -0.43 | -0.54 to -0.33 | -2.00 | 0 of 20 | 0.000 | worse | 1.00 | 0.000 (equivalent) |
| magic | -5.05 | -5.64 to -4.46 | -4.01 | 0 of 20 | 0.000 | worse | 1.00 | 1.000 |
| letter | -17.53 | -17.88 to -17.18 | -23.63 | 0 of 20 | 0.001 | worse | 1.00 | 1.000 |
| landsat | -3.69 | -4.26 to -3.13 | -3.04 | 0 of 20 | 0.001 | worse | 1.00 | 1.000 |
| page_blocks | -7.81 | -9.67 to -5.95 | -1.97 | 0 of 20 | 0.000 | worse | 1.00 | 1.000 |

**Repair at the chosen θ vs no repair there** (SR-val − B0-val; higher is better)

**SR-val is better on 3, not significantly different on 10 and worse on 0** (Holm across 13)
Equivalent within the margin on 12 of 13.

| Dataset | Difference | 95% interval | d_z | Seeds in favour | Holm p | Verdict | Margin | Equivalence Holm p |
|---|---|---|---|---|---|---|---|---|
| S1 | +0.02 | -0.01 to +0.05 | +0.32 | 4 of 20 | 1.000 | not significant | 1.00 | 0.000 (equivalent) |
| S2 | +0.01 | -0.04 to +0.05 | +0.05 | 4 of 20 | 1.000 | not significant | 1.00 | 0.000 (equivalent) |
| S3 | +0.03 | -0.04 to +0.10 | +0.20 | 4 of 20 | 1.000 | not significant | 1.00 | 0.000 (equivalent) |
| wine | +0.00 | +0.00 to +0.00 | – | 0 of 20 | 1.000 | not significant | 1.00 | 0.000 (equivalent) |
| breast_cancer | +0.09 | -0.04 to +0.21 | +0.32 | 2 of 20 | 1.000 | not significant | 1.00 | 0.000 (equivalent) |
| digits | +0.00 | +0.00 to +0.00 | – | 0 of 20 | 1.000 | not significant | 1.00 | 0.000 (equivalent) |
| thyroid | +1.55 | +1.32 to +1.78 | +3.12 | 20 of 20 | 0.001 | better | 1.00 | 1.000 |
| shuttle | +0.01 | +0.00 to +0.02 | +0.64 | 15 of 20 | 0.233 | not significant | 1.00 | 0.000 (equivalent) |
| skin | +0.14 | +0.09 to +0.18 | +1.33 | 18 of 20 | 0.000 | better | 1.00 | 0.000 (equivalent) |
| magic | -0.00 | -0.01 to +0.01 | -0.00 | 3 of 20 | 1.000 | not significant | 1.00 | 0.000 (equivalent) |
| letter | -0.00 | -0.01 to +0.01 | -0.14 | 2 of 20 | 1.000 | not significant | 1.00 | 0.000 (equivalent) |
| landsat | +0.00 | -0.00 to +0.01 | +0.22 | 1 of 20 | 1.000 | not significant | 1.00 | 0.000 (equivalent) |
| page_blocks | +0.26 | +0.12 to +0.41 | +0.83 | 15 of 20 | 0.034 | better | 1.00 | 0.000 (equivalent) |

Across datasets:

- Self-repair vs the starting model (6 datasets; positive favours repair): median +2.92 points, Wilcoxon p = 0.0312
- Repair at θ = 0.3 vs the chosen θ (13 datasets; negative favours the chosen θ): median -0.43 points, Wilcoxon p = 0.0215

## 5. Fix share against the change in accuracy (A11)

Fixed − broken is the net change in correct test predictions, so an update that does not change accuracy has a fix share near 50% by arithmetic. Means over the 10 seeds; broken is the negative-flip rate.

| Dataset | Arm | Fixed (% of test rows) | Broken (% of test rows) | Net (points) | Fix share (%) |
|---|---|---|---|---|---|
| thyroid | SR | 1.26 | 0.19 | +1.07 | 86.4 |
| thyroid | SR+AL | 1.40 | 0.19 | +1.21 | 85.3 |
| thyroid | Retrain, appended | 0.42 | 0.39 | +0.03 | 51.2 |
| thyroid | Retrain, shuffled | 1.22 | 1.03 | +0.19 | 51.8 |
| landsat | SR | 3.62 | 2.42 | +1.20 | 59.0 |
| landsat | SR+AL | 4.49 | 2.36 | +2.13 | 64.2 |
| landsat | Retrain, appended | 2.17 | 1.77 | +0.40 | 55.0 |
| landsat | Retrain, shuffled | 6.77 | 6.09 | +0.68 | 52.4 |
| page_blocks | SR | 2.76 | 0.73 | +2.03 | 69.4 |
| page_blocks | SR+AL | 5.40 | 0.82 | +4.58 | 81.9 |
| page_blocks | Retrain, appended | 2.15 | 3.23 | -1.08 | 32.0 |
| page_blocks | Retrain, shuffled | 5.20 | 6.81 | -1.61 | 42.6 |
| magic | SR | 11.15 | 7.25 | +3.91 | 60.3 |
| magic | SR+AL | 12.73 | 6.85 | +5.88 | 64.8 |
| magic | Retrain, appended | 2.61 | 2.97 | -0.36 | 46.4 |
| magic | Retrain, shuffled | 13.87 | 15.60 | -1.73 | 47.2 |
| letter | SR | 4.59 | 2.35 | +2.23 | 65.9 |
| letter | SR+AL | 6.00 | 3.06 | +2.95 | 66.2 |
| letter | Retrain, appended | 3.20 | 2.33 | +0.87 | 57.8 |
| letter | Retrain, shuffled | 10.90 | 10.48 | +0.42 | 50.9 |
| skin | SR | 3.92 | 0.54 | +3.37 | 87.5 |
| skin | SR+AL | 3.95 | 0.22 | +3.73 | 94.1 |
| skin | Retrain, appended | 0.23 | 0.49 | -0.26 | 44.4 |
| skin | Retrain, shuffled | 2.79 | 3.83 | -1.04 | 40.5 |

## 6. Faithfulness of every PACE method (C7)

Mean deletion-margin AUC in the second run (seeds 30–39); lower is more faithful. The paper's table leaves out the last three columns.

| Dataset | PACE-SHAP | KernelSHAP | LIME | PACE-flip | PACE-multi | PACE-contrastive | PACE-decomposition | Random |
|---|---|---|---|---|---|---|---|---|
| S1 | -0.1389 | -0.1326 | -0.1095 | -0.0892 | -0.0156 | 0.0474 | 0.1535 | 0.2708 |
| S2 | -0.1317 | -0.1249 | -0.0951 | -0.0755 | -0.0081 | 0.0542 | 0.1485 | 0.2565 |
| S3 | -0.0552 | -0.0412 | 0.0117 | 0.0057 | 0.0736 | 0.1191 | 0.2189 | 0.3179 |
| wine | -0.0761 | -0.0684 | -0.0387 | -0.0545 | -0.0248 | 0.0052 | 0.0396 | 0.2356 |
| breast_cancer | 0.1114 | 0.1743 | 0.1412 | 0.1366 | 0.1427 | 0.1794 | 0.2221 | 0.3981 |
| thyroid | 0.5486 | 0.5748 | 0.5708 | 0.5440 | 0.5819 | 0.5992 | 0.7230 | 0.7392 |
| shuttle | 0.3701 | 0.3652 | 0.3812 | 0.3741 | 0.3753 | 0.3931 | 0.4214 | 0.5516 |
| magic | -0.0522 | -0.0348 | -0.0115 | 0.0153 | 0.0610 | 0.0930 | 0.0937 | 0.1400 |
| letter | -0.6897 | -0.6776 | -0.6561 | -0.6441 | -0.5499 | -0.4569 | -0.5910 | -0.3811 |
| landsat | -0.3287 | -0.2166 | -0.2954 | -0.2546 | -0.2341 | -0.1540 | -0.1890 | -0.0497 |
| page_blocks | 0.0501 | 0.0464 | 0.0487 | 0.0514 | 0.0559 | 0.0765 | 0.1105 | 0.1282 |

Against the random ranking and against KernelSHAP (Holm across the 11 datasets; better / not significant / worse for the PACE method):

| Method | vs random | vs KernelSHAP |
|---|---|---|
| PACE-flip | 11 / 0 / 0 | 3 / 2 / 6 |
| PACE-multi | 11 / 0 / 0 | 1 / 4 / 6 |
| PACE-contrastive | 11 / 0 / 0 | 0 / 2 / 9 |
| PACE-decomposition | 9 / 2 / 0 | 0 / 0 / 11 |

## 7. How often a counterfactual survives a refit (A8)

Share of 4 CFMNNs, fitted on the same rows in other orders, that still give the target class at the answer; mean over points and seeds 110–119, as in the run's summary. The paper gives only the counts against DiCE's random search.

| Dataset | PACE | PACE-sparse | DiCE random | DiCE genetic | DiCE kdtree | Nearest unlike neighbour | Brute force |
|---|---|---|---|---|---|---|---|
| S1 | 0.43 | 0.33 | 0.47 | 0.90 | 1.00 | 1.00 | 0.25 |
| S2 | 0.39 | 0.35 | 0.46 | 0.90 | 1.00 | 1.00 | 0.29 |
| S3 | 0.55 | 0.39 | 0.46 | 0.92 | 1.00 | 1.00 | 0.28 |
| wine | 0.28 | 0.28 | 0.41 | 0.99 | 1.00 | 1.00 | 0.29 |
| breast_cancer | 0.17 | 0.30 | 0.38 | 0.79 | 0.78 | 0.85 | 0.27 |
| thyroid | 0.49 | 0.42 | 0.46 | 0.59 | 0.77 | 0.66 | 0.52 |
| shuttle | 0.98 | 0.98 | 0.77 | 0.86 | 0.82 | 0.99 | 0.49 |
| magic | 0.43 | 0.39 | 0.50 | 0.56 | 0.53 | 0.43 | 0.38 |
| letter | 0.27 | 0.25 | 0.28 | 0.52 | 0.49 | 0.50 | 0.23 |
| landsat | 0.23 | 0.24 | 0.35 | 0.60 | 0.55 | 0.54 | 0.27 |
| page_blocks | 0.28 | 0.41 | 0.59 | 0.54 | 0.69 | 0.32 | 0.20 |

## 8. Cost of the trust signals (C7)

Seconds per seed in the confirmatory run. The signal time covers PD, EI and U on the calibration and evaluation rows; the baseline time covers Trust Score, nearest-neighbour disagreement and the order ensemble on the evaluation rows.

The signals took longer on 14 of 16 datasets.

| Dataset | PACE signals (s) | Baselines (s) |
|---|---|---|
| S1 | 21.6 | 3.0 |
| S2 | 22.1 | 3.1 |
| S2-imbalanced | 22.5 | 3.7 |
| S2-label-noise | 22.0 | 3.2 |
| S2-shifted | 48.8 | 5.7 |
| S3 | 24.0 | 3.1 |
| breast_cancer | 0.3 | 0.2 |
| digits | 7.6 | 0.8 |
| landsat | 14.1 | 1.9 |
| letter | 87.0 | 9.1 |
| magic | 22.4 | 4.2 |
| page_blocks | 1.0 | 0.6 |
| shuttle | 7.1 | 6.5 |
| skin | 19.9 | 23.1 |
| thyroid | 3.4 | 1.1 |
| wine | 0.0 | 0.0 |

## 9. Where the base model errs (A5)

The model at θ = 0.3, means over seeds 70–89. Proposition 1 covers training errors only. Where the training error is 0, self-repair has nothing to fix, yet test errors remain.

| Dataset | Training error (%) | Boxes | Training rows per box | Test accuracy (%) | Repair edits accepted | Test accuracy after repair (%) |
|---|---|---|---|---|---|---|
| S1 | 0.0 | 1516.2 | 2.6 | 89.1 | 0.0 | 89.1 |
| S2 | 0.0 | 1533.3 | 2.6 | 80.8 | 0.2 | 80.8 |
| S3 | 0.0 | 1525.3 | 2.6 | 87.8 | 0.1 | 87.8 |
| wine | 0.0 | 31.4 | 4.5 | 98.3 | 0.0 | 98.3 |
| breast_cancer | 0.8 | 42.8 | 10.7 | 95.4 | 3.3 | 95.7 |
| digits | 0.0 | 294.4 | 4.9 | 96.0 | 0.0 | 96.0 |
| thyroid | 1.6 | 53.9 | 107.6 | 97.4 | 18.1 | 98.4 |
| shuttle | 0.2 | 13.9 | 3335.5 | 99.8 | 3.2 | 99.8 |
| skin | 4.0 | 86.7 | 2267.6 | 96.0 | 34.0 | 99.3 |
| magic | 26.6 | 285.6 | 53.4 | 71.6 | 191.2 | 76.7 |
| letter | 21.3 | 564.8 | 28.4 | 75.1 | 215.2 | 77.6 |
| landsat | 10.4 | 198.8 | 25.9 | 85.1 | 189.0 | 86.3 |
| page_blocks | 14.4 | 25.1 | 175.9 | 85.0 | 12.9 | 87.6 |

## 10. Test predictions decided by the nearest-centroid rule (A4)

The base model on the paper's 80/20 split, seeds 0–4, every test row. A row is tied when the best box of another class has the winner's membership, so the class comes from the nearest centroid, which PACE-SHAP's margin ignores.

| Dataset | Test rows | Tied (%) | Error rate, tied rows (%) | Error rate, other rows (%) | Share of all errors that are tied (%) |
|---|---|---|---|---|---|
| S1 | 5000 | 0.0 | – | 10.4 | 0.0 |
| S2 | 5000 | 0.0 | – | 19.4 | 0.0 |
| S3 | 5000 | 0.0 | – | 13.0 | 0.0 |
| wine | 180 | 0.0 | – | 1.7 | 0.0 |
| breast_cancer | 570 | 0.4 | 0.0 | 4.6 | 0.0 |
| digits | 1800 | 0.7 | 50.0 | 4.4 | 7.1 |
| thyroid | 7200 | 5.3 | 32.1 | 1.2 | 59.8 |
| shuttle | 58000 | 12.6 | 1.3 | 0.0 | 83.1 |
| skin | 245060 | 20.2 | 26.0 | 0.0 | 99.9 |
| magic | 19020 | 76.2 | 32.6 | 11.2 | 90.3 |
| letter | 20000 | 56.5 | 40.7 | 5.5 | 90.6 |
| landsat | 6435 | 24.7 | 33.9 | 9.7 | 53.4 |
| page_blocks | 5475 | 72.8 | 17.9 | 2.6 | 94.9 |
