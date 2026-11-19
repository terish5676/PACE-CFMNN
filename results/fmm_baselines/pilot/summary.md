# Run P2: CFMNN with self-repair against tuned fuzzy min-max networks

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The paper split, seeds 194–194 (1); theta by 3-fold cross-validation for every model. Written by `scripts/run_fmm_baselines.py`.

## Primary: test macro-F1, CFMNN-SR-val against each tuned baseline

Holm across all 24 tests. better 0, no significant difference 24, worse 0. CFMNN with repair is never significantly worse than a tuned fuzzy min-max baseline.

| Baseline | Dataset | Seeds | CFMNN-SR-val | Baseline | Difference | Seeds better / worse | p | p (Holm) | Verdict for CFMNN-SR-val |
|---|---|---|---|---|---|---|---|---|---|
| EFMNN-val | thyroid | 1 | 95.30 | 70.19 | +25.11 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| EFMNN-val | landsat | 1 | 87.61 | 87.59 | +0.01 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| EFMNN-val | page_blocks | 1 | 85.49 | 85.02 | +0.47 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| EFMNN-val | magic | 1 | 80.11 | 78.27 | +1.85 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| EFMNN-val | letter | 1 | 95.81 | 95.81 | -0.00 | 0 / 1 | 1.0000 | 1.0000 | no significant difference |
| EFMNN-val | skin | 1 | 99.90 | 99.87 | +0.03 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| KNEFMNN-val | thyroid | 1 | 95.30 | 78.17 | +17.13 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| KNEFMNN-val | landsat | 1 | 87.61 | 87.76 | -0.15 | 0 / 1 | 1.0000 | 1.0000 | no significant difference |
| KNEFMNN-val | page_blocks | 1 | 85.49 | 71.87 | +13.62 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| KNEFMNN-val | magic | 1 | 80.11 | 80.14 | -0.03 | 0 / 1 | 1.0000 | 1.0000 | no significant difference |
| KNEFMNN-val | letter | 1 | 95.81 | 95.81 | -0.00 | 0 / 1 | 1.0000 | 1.0000 | no significant difference |
| KNEFMNN-val | skin | 1 | 99.90 | 99.82 | +0.08 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| RFMNN-val | thyroid | 1 | 95.30 | 84.86 | +10.43 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| RFMNN-val | landsat | 1 | 87.61 | 87.59 | +0.01 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| RFMNN-val | page_blocks | 1 | 85.49 | 85.18 | +0.31 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| RFMNN-val | magic | 1 | 80.11 | 78.27 | +1.85 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| RFMNN-val | letter | 1 | 95.81 | 95.81 | -0.00 | 0 / 1 | 1.0000 | 1.0000 | no significant difference |
| RFMNN-val | skin | 1 | 99.90 | 99.88 | +0.02 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| IOL-GFMM-val | thyroid | 1 | 95.30 | 86.86 | +8.44 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| IOL-GFMM-val | landsat | 1 | 87.61 | 87.75 | -0.14 | 0 / 1 | 1.0000 | 1.0000 | no significant difference |
| IOL-GFMM-val | page_blocks | 1 | 85.49 | 72.89 | +12.61 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| IOL-GFMM-val | magic | 1 | 80.11 | 82.32 | -2.20 | 0 / 1 | 1.0000 | 1.0000 | no significant difference |
| IOL-GFMM-val | letter | 1 | 95.81 | 91.63 | +4.18 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |
| IOL-GFMM-val | skin | 1 | 99.90 | 99.87 | +0.03 | 1 / 0 | 1.0000 | 1.0000 | no significant difference |

## Secondary: test accuracy, tuned

CFMNN-SR-val against each; Holm across 24 tests. better 0, no significant difference 24, worse 0.

| Baseline | Dataset | Difference | Seeds better / worse | p (Holm) | Verdict |
|---|---|---|---|---|---|
| EFMNN-val | thyroid | +4.93 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-val | landsat | +0.47 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-val | page_blocks | -1.00 | 0 / 1 | 1.0000 | no significant difference |
| EFMNN-val | magic | +1.68 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-val | letter | -0.03 | 0 / 1 | 1.0000 | no significant difference |
| EFMNN-val | skin | +0.02 | 1 / 0 | 1.0000 | no significant difference |
| KNEFMNN-val | thyroid | +3.12 | 1 / 0 | 1.0000 | no significant difference |
| KNEFMNN-val | landsat | +0.31 | 1 / 0 | 1.0000 | no significant difference |
| KNEFMNN-val | page_blocks | +0.27 | 1 / 0 | 1.0000 | no significant difference |
| KNEFMNN-val | magic | -0.13 | 0 / 1 | 1.0000 | no significant difference |
| KNEFMNN-val | letter | -0.03 | 0 / 1 | 1.0000 | no significant difference |
| KNEFMNN-val | skin | +0.05 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-val | thyroid | +1.88 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-val | landsat | +0.47 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-val | page_blocks | -1.37 | 0 / 1 | 1.0000 | no significant difference |
| RFMNN-val | magic | +1.68 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-val | letter | -0.03 | 0 / 1 | 1.0000 | no significant difference |
| RFMNN-val | skin | +0.01 | 1 / 0 | 1.0000 | no significant difference |
| IOL-GFMM-val | thyroid | +1.39 | 1 / 0 | 1.0000 | no significant difference |
| IOL-GFMM-val | landsat | -0.54 | 0 / 1 | 1.0000 | no significant difference |
| IOL-GFMM-val | page_blocks | +0.00 | 0 / 0 | 1.0000 | no significant difference |
| IOL-GFMM-val | magic | -2.13 | 0 / 1 | 1.0000 | no significant difference |
| IOL-GFMM-val | letter | +4.17 | 1 / 0 | 1.0000 | no significant difference |
| IOL-GFMM-val | skin | +0.02 | 1 / 0 | 1.0000 | no significant difference |

## Secondary: test macro-F1 at a fixed theta = 0.3

CFMNN-SR-0.3 against each; Holm across 24 tests. better 0, no significant difference 24, worse 0.

| Baseline | Dataset | Difference | Seeds better / worse | p (Holm) | Verdict |
|---|---|---|---|---|---|
| EFMNN-0.3 | thyroid | +22.31 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-0.3 | landsat | +1.62 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-0.3 | page_blocks | -11.78 | 0 / 1 | 1.0000 | no significant difference |
| EFMNN-0.3 | magic | +9.89 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-0.3 | letter | -1.79 | 0 / 1 | 1.0000 | no significant difference |
| EFMNN-0.3 | skin | +1.21 | 1 / 0 | 1.0000 | no significant difference |
| KNEFMNN-0.3 | thyroid | +12.72 | 1 / 0 | 1.0000 | no significant difference |
| KNEFMNN-0.3 | landsat | -0.20 | 0 / 1 | 1.0000 | no significant difference |
| KNEFMNN-0.3 | page_blocks | -8.12 | 0 / 1 | 1.0000 | no significant difference |
| KNEFMNN-0.3 | magic | +2.87 | 1 / 0 | 1.0000 | no significant difference |
| KNEFMNN-0.3 | letter | -7.11 | 0 / 1 | 1.0000 | no significant difference |
| KNEFMNN-0.3 | skin | +1.57 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-0.3 | thyroid | +11.92 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-0.3 | landsat | +0.67 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-0.3 | page_blocks | -15.32 | 0 / 1 | 1.0000 | no significant difference |
| RFMNN-0.3 | magic | -3.69 | 0 / 1 | 1.0000 | no significant difference |
| RFMNN-0.3 | letter | -6.72 | 0 / 1 | 1.0000 | no significant difference |
| RFMNN-0.3 | skin | -0.30 | 0 / 1 | 1.0000 | no significant difference |
| IOL-GFMM-0.3 | thyroid | +7.32 | 1 / 0 | 1.0000 | no significant difference |
| IOL-GFMM-0.3 | landsat | -3.39 | 0 / 1 | 1.0000 | no significant difference |
| IOL-GFMM-0.3 | page_blocks | -17.16 | 0 / 1 | 1.0000 | no significant difference |
| IOL-GFMM-0.3 | magic | -7.46 | 0 / 1 | 1.0000 | no significant difference |
| IOL-GFMM-0.3 | letter | -9.49 | 0 / 1 | 1.0000 | no significant difference |
| IOL-GFMM-0.3 | skin | +0.13 | 1 / 0 | 1.0000 | no significant difference |

## Descriptive: test macro-F1, tuned CFMNN without repair

CFMNN-val against each; Holm across 24 tests. better 0, no significant difference 24, worse 0.

| Baseline | Dataset | Difference | Seeds better / worse | p (Holm) | Verdict |
|---|---|---|---|---|---|
| EFMNN-val | thyroid | +19.02 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-val | landsat | +0.01 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-val | page_blocks | +0.32 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-val | magic | +1.81 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-val | letter | +0.05 | 1 / 0 | 1.0000 | no significant difference |
| EFMNN-val | skin | -0.12 | 0 / 1 | 1.0000 | no significant difference |
| KNEFMNN-val | thyroid | +11.04 | 1 / 0 | 1.0000 | no significant difference |
| KNEFMNN-val | landsat | -0.15 | 0 / 1 | 1.0000 | no significant difference |
| KNEFMNN-val | page_blocks | +13.47 | 1 / 0 | 1.0000 | no significant difference |
| KNEFMNN-val | magic | -0.06 | 0 / 1 | 1.0000 | no significant difference |
| KNEFMNN-val | letter | +0.05 | 1 / 0 | 1.0000 | no significant difference |
| KNEFMNN-val | skin | -0.07 | 0 / 1 | 1.0000 | no significant difference |
| RFMNN-val | thyroid | +4.35 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-val | landsat | +0.01 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-val | page_blocks | +0.15 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-val | magic | +1.81 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-val | letter | +0.05 | 1 / 0 | 1.0000 | no significant difference |
| RFMNN-val | skin | -0.13 | 0 / 1 | 1.0000 | no significant difference |
| IOL-GFMM-val | thyroid | +2.36 | 1 / 0 | 1.0000 | no significant difference |
| IOL-GFMM-val | landsat | -0.14 | 0 / 1 | 1.0000 | no significant difference |
| IOL-GFMM-val | page_blocks | +12.45 | 1 / 0 | 1.0000 | no significant difference |
| IOL-GFMM-val | magic | -2.24 | 0 / 1 | 1.0000 | no significant difference |
| IOL-GFMM-val | letter | +4.23 | 1 / 0 | 1.0000 | no significant difference |
| IOL-GFMM-val | skin | -0.12 | 0 / 1 | 1.0000 | no significant difference |

## Test macro-F1 (%), mean over seeds

| Dataset | CFMNN-SR-val | CFMNN-val | EFMNN-val | KNEFMNN-val | RFMNN-val | IOL-GFMM-val | CFMNN-SR-0.3 | CFMNN-0.3 | EFMNN-0.3 | KNEFMNN-0.3 | RFMNN-0.3 | IOL-GFMM-0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thyroid | 95.30 | 89.21 | 70.19 | 78.17 | 84.86 | 86.86 | 92.13 | 83.88 | 69.82 | 79.41 | 80.21 | 84.81 |
| landsat | 87.61 | 87.61 | 87.59 | 87.76 | 87.59 | 87.75 | 84.35 | 81.19 | 82.74 | 84.55 | 83.68 | 87.75 |
| page_blocks | 85.49 | 85.33 | 85.02 | 71.87 | 85.18 | 72.89 | 59.05 | 57.30 | 70.83 | 67.17 | 74.37 | 76.21 |
| magic | 80.11 | 80.08 | 78.27 | 80.14 | 78.27 | 82.32 | 74.86 | 69.87 | 64.97 | 71.99 | 78.55 | 82.32 |
| letter | 95.81 | 95.86 | 95.81 | 95.81 | 95.81 | 91.63 | 77.51 | 74.10 | 79.30 | 84.62 | 84.23 | 87.00 |
| skin | 99.90 | 99.75 | 99.87 | 99.82 | 99.88 | 99.87 | 99.09 | 95.56 | 97.89 | 97.52 | 99.39 | 98.97 |

## Boxes, mean over seeds

| Dataset | CFMNN-SR-val | CFMNN-val | EFMNN-val | KNEFMNN-val | RFMNN-val | IOL-GFMM-val | CFMNN-SR-0.3 | CFMNN-0.3 | EFMNN-0.3 | KNEFMNN-0.3 | RFMNN-0.3 | IOL-GFMM-0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thyroid | 15 | 15 | 852 | 316 | 723 | 276 | 51 | 52 | 1211 | 408 | 1244 | 387 |
| landsat | 2521 | 2521 | 5138 | 5133 | 5138 | 543 | 204 | 204 | 2356 | 764 | 3004 | 543 |
| page_blocks | 311 | 313 | 1970 | 1051 | 1997 | 156 | 26 | 26 | 514 | 96 | 739 | 181 |
| magic | 5763 | 5763 | 13006 | 11099 | 13006 | 2221 | 286 | 286 | 6958 | 1510 | 9241 | 2221 |
| letter | 5237 | 5237 | 15069 | 15069 | 15069 | 15069 | 545 | 545 | 5511 | 1158 | 7357 | 1906 |
| skin | 2017 | 2020 | 11212 | 2333 | 11288 | 2075 | 79 | 80 | 878 | 76 | 8154 | 1012 |

## Seconds (tuning + fit + repair), mean over seeds

| Dataset | CFMNN-SR-val | CFMNN-val | EFMNN-val | KNEFMNN-val | RFMNN-val | IOL-GFMM-val | CFMNN-SR-0.3 | CFMNN-0.3 | EFMNN-0.3 | KNEFMNN-0.3 | RFMNN-0.3 | IOL-GFMM-0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thyroid | 5 | 4 | 160 | 128 | 31 | 31 | 2 | 0 | 12 | 6 | 1 | 1 |
| landsat | 43 | 14 | 373 | 384 | 103 | 87 | 90 | 0 | 35 | 24 | 3 | 4 |
| page_blocks | 17 | 3 | 94 | 75 | 9 | 7 | 9 | 0 | 3 | 2 | 0 | 0 |
| magic | 177 | 40 | 3040 | 2590 | 247 | 808 | 142 | 1 | 244 | 87 | 9 | 89 |
| letter | 570 | 61 | 4577 | 3552 | 583 | 448 | 712 | 0 | 238 | 110 | 9 | 17 |
| skin | 1897 | 176 | 3544 | 813 | 1247 | 749 | 103 | 6 | 86 | 18 | 49 | 39 |

## Theta chosen by cross-validation (count over seeds)

- CFMNN: thyroid 0.5 ×1; landsat 0.05 ×1; page_blocks 0.05 ×1; magic 0.05 ×1; letter 0.05 ×1; skin 0.05 ×1
- EFMNN: thyroid 0.4 ×1; landsat 0.05 ×1; page_blocks 0.05 ×1; magic 0.05 ×1; letter 0.05 ×1; skin 0.05 ×1
- KNEFMNN: thyroid 0.4 ×1; landsat 0.05 ×1; page_blocks 0.05 ×1; magic 0.05 ×1; letter 0.05 ×1; skin 0.05 ×1
- RFMNN: thyroid 0.5 ×1; landsat 0.05 ×1; page_blocks 0.05 ×1; magic 0.05 ×1; letter 0.05 ×1; skin 0.05 ×1
- IOL-GFMM: thyroid 0.5 ×1; landsat 0.3 ×1; page_blocks 0.4 ×1; magic 0.3 ×1; letter 0.05 ×1; skin 0.05 ×1
