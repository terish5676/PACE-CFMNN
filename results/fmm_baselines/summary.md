# Run P2: CFMNN with self-repair against tuned fuzzy min-max networks

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The paper split, seeds 250–259 (10); theta by 3-fold cross-validation for every model. Written by `scripts/run_fmm_baselines.py`.

## Primary: test macro-F1, CFMNN-SR-val against each tuned baseline

Holm across all 24 tests. better 6, no significant difference 17, worse 1. Mixed: CFMNN with repair is significantly worse in 1 of 24 comparisons; the paper names them.

| Baseline | Dataset | Seeds | CFMNN-SR-val | Baseline | Difference | Seeds better / worse | p | p (Holm) | Verdict for CFMNN-SR-val |
|---|---|---|---|---|---|---|---|---|---|
| EFMNN-val | thyroid | 10 | 94.28 | 64.88 | +29.40 | 10 / 0 | 0.0020 | 0.0469 | better |
| EFMNN-val | landsat | 10 | 88.73 | 88.65 | +0.08 | 4 / 6 | 1.0000 | 1.0000 | no significant difference |
| EFMNN-val | page_blocks | 10 | 78.69 | 78.38 | +0.31 | 4 / 6 | 1.0000 | 1.0000 | no significant difference |
| EFMNN-val | magic | 10 | 79.81 | 79.34 | +0.48 | 9 / 1 | 0.0273 | 0.4648 | no significant difference |
| EFMNN-val | letter | 10 | 94.84 | 95.06 | -0.22 | 3 / 7 | 0.0645 | 0.8379 | no significant difference |
| EFMNN-val | skin | 10 | 99.89 | 99.89 | +0.01 | 8 / 2 | 0.1309 | 1.0000 | no significant difference |
| KNEFMNN-val | thyroid | 10 | 94.28 | 72.65 | +21.63 | 10 / 0 | 0.0020 | 0.0469 | better |
| KNEFMNN-val | landsat | 10 | 88.73 | 88.08 | +0.65 | 7 / 3 | 0.0273 | 0.4648 | no significant difference |
| KNEFMNN-val | page_blocks | 10 | 78.69 | 75.99 | +2.71 | 6 / 4 | 0.3750 | 1.0000 | no significant difference |
| KNEFMNN-val | magic | 10 | 79.81 | 79.62 | +0.19 | 7 / 3 | 0.2324 | 1.0000 | no significant difference |
| KNEFMNN-val | letter | 10 | 94.84 | 95.06 | -0.22 | 3 / 7 | 0.0645 | 0.8379 | no significant difference |
| KNEFMNN-val | skin | 10 | 99.89 | 99.88 | +0.01 | 6 / 4 | 0.2754 | 1.0000 | no significant difference |
| RFMNN-val | thyroid | 10 | 94.28 | 79.41 | +14.87 | 10 / 0 | 0.0020 | 0.0469 | better |
| RFMNN-val | landsat | 10 | 88.73 | 88.65 | +0.08 | 4 / 6 | 1.0000 | 1.0000 | no significant difference |
| RFMNN-val | page_blocks | 10 | 78.69 | 79.30 | -0.60 | 4 / 6 | 0.6250 | 1.0000 | no significant difference |
| RFMNN-val | magic | 10 | 79.81 | 79.32 | +0.49 | 9 / 1 | 0.0273 | 0.4648 | no significant difference |
| RFMNN-val | letter | 10 | 94.84 | 95.06 | -0.22 | 3 / 7 | 0.0645 | 0.8379 | no significant difference |
| RFMNN-val | skin | 10 | 99.89 | 99.92 | -0.02 | 2 / 8 | 0.0488 | 0.6836 | no significant difference |
| IOL-GFMM-val | thyroid | 10 | 94.28 | 79.69 | +14.59 | 10 / 0 | 0.0020 | 0.0469 | better |
| IOL-GFMM-val | landsat | 10 | 88.73 | 86.40 | +2.34 | 10 / 0 | 0.0020 | 0.0469 | better |
| IOL-GFMM-val | page_blocks | 10 | 78.69 | 74.69 | +4.01 | 8 / 2 | 0.0645 | 0.8379 | no significant difference |
| IOL-GFMM-val | magic | 10 | 79.81 | 81.04 | -1.22 | 0 / 10 | 0.0020 | 0.0469 | worse |
| IOL-GFMM-val | letter | 10 | 94.84 | 91.67 | +3.17 | 10 / 0 | 0.0020 | 0.0469 | better |
| IOL-GFMM-val | skin | 10 | 99.89 | 99.89 | +0.00 | 6 / 4 | 0.6953 | 1.0000 | no significant difference |

## Secondary: test accuracy, tuned

CFMNN-SR-val against each; Holm across 24 tests. better 6, no significant difference 17, worse 1.

| Baseline | Dataset | Difference | Seeds better / worse | p (Holm) | Verdict |
|---|---|---|---|---|---|
| EFMNN-val | thyroid | +4.80 | 10 / 0 | 0.0469 | better |
| EFMNN-val | landsat | +0.22 | 5 / 5 | 1.0000 | no significant difference |
| EFMNN-val | page_blocks | -0.20 | 3 / 7 | 1.0000 | no significant difference |
| EFMNN-val | magic | +0.30 | 8 / 2 | 0.8750 | no significant difference |
| EFMNN-val | letter | -0.22 | 3 / 7 | 0.8750 | no significant difference |
| EFMNN-val | skin | +0.00 | 8 / 2 | 0.8789 | no significant difference |
| KNEFMNN-val | thyroid | +3.72 | 10 / 0 | 0.0469 | better |
| KNEFMNN-val | landsat | +0.82 | 8 / 1 | 0.1328 | no significant difference |
| KNEFMNN-val | page_blocks | -0.21 | 2 / 8 | 0.8789 | no significant difference |
| KNEFMNN-val | magic | -0.12 | 5 / 5 | 1.0000 | no significant difference |
| KNEFMNN-val | letter | -0.22 | 3 / 7 | 0.8750 | no significant difference |
| KNEFMNN-val | skin | +0.01 | 6 / 4 | 1.0000 | no significant difference |
| RFMNN-val | thyroid | +2.61 | 10 / 0 | 0.0469 | better |
| RFMNN-val | landsat | +0.22 | 5 / 5 | 1.0000 | no significant difference |
| RFMNN-val | page_blocks | -0.43 | 1 / 8 | 0.8203 | no significant difference |
| RFMNN-val | magic | +0.31 | 8 / 2 | 0.8750 | no significant difference |
| RFMNN-val | letter | -0.22 | 3 / 7 | 0.8750 | no significant difference |
| RFMNN-val | skin | -0.02 | 2 / 8 | 0.6250 | no significant difference |
| IOL-GFMM-val | thyroid | +2.33 | 10 / 0 | 0.0469 | better |
| IOL-GFMM-val | landsat | +1.63 | 10 / 0 | 0.0469 | better |
| IOL-GFMM-val | page_blocks | -0.32 | 2 / 8 | 1.0000 | no significant difference |
| IOL-GFMM-val | magic | -1.30 | 0 / 10 | 0.0469 | worse |
| IOL-GFMM-val | letter | +3.18 | 10 / 0 | 0.0469 | better |
| IOL-GFMM-val | skin | +0.00 | 6 / 4 | 1.0000 | no significant difference |

## Secondary: test macro-F1 at a fixed theta = 0.3

CFMNN-SR-0.3 against each; Holm across 24 tests. better 7, no significant difference 11, worse 6.

| Baseline | Dataset | Difference | Seeds better / worse | p (Holm) | Verdict |
|---|---|---|---|---|---|
| EFMNN-0.3 | thyroid | +26.34 | 10 / 0 | 0.0469 | better |
| EFMNN-0.3 | landsat | +2.59 | 9 / 1 | 0.0469 | better |
| EFMNN-0.3 | page_blocks | -0.85 | 3 / 7 | 1.0000 | no significant difference |
| EFMNN-0.3 | magic | +5.64 | 8 / 2 | 0.1074 | no significant difference |
| EFMNN-0.3 | letter | -0.11 | 4 / 6 | 1.0000 | no significant difference |
| EFMNN-0.3 | skin | -0.53 | 1 / 9 | 0.2188 | no significant difference |
| KNEFMNN-0.3 | thyroid | +22.66 | 10 / 0 | 0.0469 | better |
| KNEFMNN-0.3 | landsat | +0.16 | 5 / 5 | 1.0000 | no significant difference |
| KNEFMNN-0.3 | page_blocks | -0.79 | 6 / 4 | 1.0000 | no significant difference |
| KNEFMNN-0.3 | magic | +3.25 | 10 / 0 | 0.0469 | better |
| KNEFMNN-0.3 | letter | -6.07 | 0 / 10 | 0.0469 | worse |
| KNEFMNN-0.3 | skin | +0.81 | 10 / 0 | 0.0469 | better |
| RFMNN-0.3 | thyroid | +17.67 | 10 / 0 | 0.0469 | better |
| RFMNN-0.3 | landsat | +1.34 | 8 / 2 | 0.2188 | no significant difference |
| RFMNN-0.3 | page_blocks | -9.93 | 2 / 8 | 0.1367 | no significant difference |
| RFMNN-0.3 | magic | -1.24 | 2 / 8 | 0.2188 | no significant difference |
| RFMNN-0.3 | letter | -7.08 | 0 / 10 | 0.0469 | worse |
| RFMNN-0.3 | skin | -0.68 | 0 / 10 | 0.0469 | worse |
| IOL-GFMM-0.3 | thyroid | +14.45 | 10 / 0 | 0.0469 | better |
| IOL-GFMM-0.3 | landsat | -2.74 | 0 / 10 | 0.0469 | worse |
| IOL-GFMM-0.3 | page_blocks | -9.65 | 1 / 9 | 0.1367 | no significant difference |
| IOL-GFMM-0.3 | magic | -6.33 | 0 / 10 | 0.0469 | worse |
| IOL-GFMM-0.3 | letter | -9.68 | 0 / 10 | 0.0469 | worse |
| IOL-GFMM-0.3 | skin | -0.17 | 3 / 7 | 1.0000 | no significant difference |

## Descriptive: test macro-F1, tuned CFMNN without repair

CFMNN-val against each; Holm across 24 tests. better 5, no significant difference 15, worse 4.

| Baseline | Dataset | Difference | Seeds better / worse | p (Holm) | Verdict |
|---|---|---|---|---|---|
| EFMNN-val | thyroid | +22.86 | 10 / 0 | 0.0469 | better |
| EFMNN-val | landsat | +0.08 | 4 / 6 | 1.0000 | no significant difference |
| EFMNN-val | page_blocks | -2.34 | 2 / 8 | 0.9609 | no significant difference |
| EFMNN-val | magic | +0.47 | 9 / 1 | 0.3555 | no significant difference |
| EFMNN-val | letter | -0.22 | 3 / 7 | 0.6445 | no significant difference |
| EFMNN-val | skin | -0.20 | 0 / 10 | 0.0469 | worse |
| KNEFMNN-val | thyroid | +15.09 | 10 / 0 | 0.0469 | better |
| KNEFMNN-val | landsat | +0.65 | 7 / 3 | 0.3555 | no significant difference |
| KNEFMNN-val | page_blocks | +0.05 | 5 / 5 | 1.0000 | no significant difference |
| KNEFMNN-val | magic | +0.18 | 7 / 3 | 1.0000 | no significant difference |
| KNEFMNN-val | letter | -0.22 | 3 / 7 | 0.6445 | no significant difference |
| KNEFMNN-val | skin | -0.19 | 0 / 10 | 0.0469 | worse |
| RFMNN-val | thyroid | +8.33 | 9 / 1 | 0.0586 | no significant difference |
| RFMNN-val | landsat | +0.08 | 4 / 6 | 1.0000 | no significant difference |
| RFMNN-val | page_blocks | -3.26 | 2 / 8 | 0.7383 | no significant difference |
| RFMNN-val | magic | +0.49 | 9 / 1 | 0.3555 | no significant difference |
| RFMNN-val | letter | -0.22 | 3 / 7 | 0.6445 | no significant difference |
| RFMNN-val | skin | -0.23 | 0 / 10 | 0.0469 | worse |
| IOL-GFMM-val | thyroid | +8.05 | 10 / 0 | 0.0469 | better |
| IOL-GFMM-val | landsat | +2.34 | 10 / 0 | 0.0469 | better |
| IOL-GFMM-val | page_blocks | +1.35 | 7 / 3 | 1.0000 | no significant difference |
| IOL-GFMM-val | magic | -1.23 | 0 / 10 | 0.0469 | worse |
| IOL-GFMM-val | letter | +3.17 | 10 / 0 | 0.0469 | better |
| IOL-GFMM-val | skin | -0.20 | 1 / 9 | 0.0586 | no significant difference |

## Test macro-F1 (%), mean over seeds

| Dataset | CFMNN-SR-val | CFMNN-val | EFMNN-val | KNEFMNN-val | RFMNN-val | IOL-GFMM-val | CFMNN-SR-0.3 | CFMNN-0.3 | EFMNN-0.3 | KNEFMNN-0.3 | RFMNN-0.3 | IOL-GFMM-0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thyroid | 94.28 | 87.74 | 64.88 | 72.65 | 79.41 | 79.69 | 92.43 | 84.97 | 66.09 | 69.76 | 74.76 | 77.98 |
| landsat | 88.73 | 88.73 | 88.65 | 88.08 | 88.65 | 86.40 | 83.27 | 81.95 | 80.69 | 83.11 | 81.94 | 86.01 |
| page_blocks | 78.69 | 76.04 | 78.38 | 75.99 | 79.30 | 74.69 | 64.24 | 60.81 | 65.09 | 65.02 | 74.17 | 73.88 |
| magic | 79.81 | 79.81 | 79.34 | 79.62 | 79.32 | 81.04 | 74.69 | 71.43 | 69.05 | 71.43 | 75.93 | 81.01 |
| letter | 94.84 | 94.84 | 95.06 | 95.06 | 95.06 | 91.67 | 77.46 | 75.19 | 77.57 | 83.53 | 84.54 | 87.14 |
| skin | 99.89 | 99.69 | 99.89 | 99.88 | 99.92 | 99.89 | 98.86 | 89.60 | 99.38 | 98.04 | 99.53 | 99.02 |

## Boxes, mean over seeds

| Dataset | CFMNN-SR-val | CFMNN-val | EFMNN-val | KNEFMNN-val | RFMNN-val | IOL-GFMM-val | CFMNN-SR-0.3 | CFMNN-0.3 | EFMNN-0.3 | KNEFMNN-0.3 | RFMNN-0.3 | IOL-GFMM-0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thyroid | 17 | 18 | 743 | 528 | 787 | 300 | 53 | 53 | 1170 | 413 | 1210 | 387 |
| landsat | 2299 | 2299 | 5139 | 5134 | 5139 | 994 | 204 | 204 | 2412 | 752 | 3105 | 549 |
| page_blocks | 327 | 328 | 2024 | 1008 | 2056 | 253 | 25 | 26 | 580 | 96 | 918 | 188 |
| magic | 5747 | 5747 | 12986 | 11095 | 12988 | 2340 | 290 | 290 | 6332 | 1589 | 9492 | 2282 |
| letter | 5343 | 5343 | 15075 | 15075 | 15075 | 15075 | 568 | 568 | 5512 | 1171 | 7158 | 1843 |
| skin | 2010 | 2012 | 10998 | 2329 | 11116 | 2087 | 89 | 89 | 1197 | 78 | 6228 | 849 |

## Seconds (tuning + fit + repair), mean over seeds

| Dataset | CFMNN-SR-val | CFMNN-val | EFMNN-val | KNEFMNN-val | RFMNN-val | IOL-GFMM-val | CFMNN-SR-0.3 | CFMNN-0.3 | EFMNN-0.3 | KNEFMNN-0.3 | RFMNN-0.3 | IOL-GFMM-0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thyroid | 7 | 5 | 161 | 134 | 36 | 34 | 4 | 0 | 11 | 6 | 1 | 1 |
| landsat | 42 | 17 | 365 | 377 | 104 | 88 | 107 | 0 | 43 | 25 | 4 | 5 |
| page_blocks | 18 | 3 | 96 | 72 | 13 | 12 | 8 | 0 | 5 | 2 | 0 | 0 |
| magic | 239 | 40 | 3487 | 2768 | 272 | 855 | 157 | 1 | 450 | 150 | 12 | 103 |
| letter | 360 | 72 | 5488 | 3598 | 556 | 487 | 816 | 1 | 457 | 187 | 10 | 17 |
| skin | 1787 | 179 | 3615 | 825 | 1177 | 1010 | 87 | 6 | 117 | 18 | 43 | 33 |

## Theta chosen by cross-validation (count over seeds)

- CFMNN: thyroid 0.3 ×1, 0.5 ×9; landsat 0.05 ×8, 0.1 ×2; page_blocks 0.05 ×10; magic 0.05 ×10; letter 0.05 ×10; skin 0.05 ×10
- EFMNN: thyroid 0.4 ×5, 0.5 ×5; landsat 0.05 ×10; page_blocks 0.05 ×9, 0.1 ×1; magic 0.05 ×10; letter 0.05 ×10; skin 0.05 ×10
- KNEFMNN: thyroid 0.2 ×7, 0.3 ×1, 0.4 ×1, 0.5 ×1; landsat 0.05 ×10; page_blocks 0.05 ×9, 0.1 ×1; magic 0.05 ×10; letter 0.05 ×10; skin 0.05 ×10
- RFMNN: thyroid 0.4 ×2, 0.5 ×8; landsat 0.05 ×10; page_blocks 0.05 ×9, 0.1 ×1; magic 0.05 ×10; letter 0.05 ×10; skin 0.05 ×10
- IOL-GFMM: thyroid 0.4 ×4, 0.5 ×6; landsat 0.1 ×1, 0.2 ×6, 0.3 ×3; page_blocks 0.05 ×1, 0.2 ×1, 0.3 ×2, 0.4 ×2, 0.5 ×4; magic 0.15 ×1, 0.2 ×2, 0.3 ×5, 0.4 ×2; letter 0.05 ×10; skin 0.05 ×10
