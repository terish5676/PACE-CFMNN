# PACE-CFMNN v6.0 Comprehensive Empirical Evaluation

**Date:** 2026-10-28 18:11:53
**Environment:** Python 3.11.0 on Windows (AMD64)

## Table A: Benchmark Inventory

| Dataset   | Task           |     N |   d |   Classes | Imbalance_Ratio   | Role                   |
|:----------|:---------------|------:|----:|----------:|:------------------|:-----------------------|
| skin      | Classification | 20000 |   3 |         2 | 3.78:1            | Benchmark / Evaluation |
| thyroid   | Classification |  7200 |  21 |         3 | 40.16:1           | Benchmark / Evaluation |
| shuttle   | Classification | 20000 |   9 |         7 | 3935.50:1         | Benchmark / Evaluation |
| digits    | Classification |  1797 |  64 |        10 | 1.05:1            | Benchmark / Evaluation |

## Table B: Predictive Baselines (Mean ± Std over Seeds)

| Dataset | Model | Accuracy (%) | Macro-F1 (%) | Balanced Acc (%) | Train (s) | Test (s) |
|---|---|---|---|---|---|---|
| digits | CFMNN+Contraction | 96.11 ± 0.56 | 96.09 ± 0.57 | 96.09 ± 0.57 | 0.050 | 0.052 |
| digits | CFMNN-fixed | 78.70 ± 1.53 | 78.10 ± 1.83 | 78.67 ± 1.48 | 0.081 | 0.412 |
| digits | CFMNN-paper | 78.70 ± 1.53 | 78.10 ± 1.83 | 78.67 ± 1.48 | 0.091 | 0.321 |
| digits | CFMNN-primary (ours) | 96.11 ± 0.56 | 96.09 ± 0.57 | 96.09 ± 0.57 | 0.041 | 0.049 |
| digits | LightGBM | 96.85 ± 0.16 | 96.85 ± 0.17 | 96.85 ± 0.16 | 0.463 | 0.009 |
| digits | LogisticRegression | 96.94 ± 0.28 | 96.94 ± 0.26 | 96.93 ± 0.29 | 0.049 | 0.000 |
| digits | RandomForest | 97.31 ± 0.16 | 97.31 ± 0.16 | 97.31 ± 0.17 | 0.246 | 0.037 |
| digits | XGBoost | 96.20 ± 0.32 | 96.20 ± 0.32 | 96.19 ± 0.33 | 0.530 | 0.004 |
| shuttle | CFMNN+Contraction | 30.77 ± 10.34 | 36.80 ± 3.17 | 57.06 ± 5.72 | 0.236 | 0.007 |
| shuttle | CFMNN-fixed | 98.63 ± 0.58 | 71.29 ± 6.35 | 67.10 ± 5.38 | 0.227 | 0.010 |
| shuttle | CFMNN-paper | 93.12 ± 0.43 | 67.95 ± 6.50 | 62.67 ± 5.44 | 0.226 | 0.011 |
| shuttle | CFMNN-primary (ours) | 98.62 ± 0.59 | 71.28 ± 6.37 | 67.08 ± 5.42 | 0.223 | 0.007 |
| shuttle | LightGBM | 94.07 ± 2.18 | 36.73 ± 4.50 | 36.21 ± 5.16 | 0.378 | 0.013 |
| shuttle | LogisticRegression | 94.90 ± 0.29 | 39.85 ± 0.19 | 39.12 ± 0.38 | 0.168 | 0.000 |
| shuttle | RandomForest | 99.92 ± 0.05 | 79.09 ± 7.39 | 78.96 ± 7.19 | 0.377 | 0.045 |
| shuttle | XGBoost | 99.97 ± 0.04 | 99.60 ± 0.35 | 99.24 ± 0.66 | 0.510 | 0.009 |
| skin | CFMNN+Contraction | 89.57 ± 2.68 | 80.19 ± 6.12 | 75.35 ± 6.22 | 0.312 | 0.020 |
| skin | CFMNN-fixed | 98.75 ± 0.37 | 98.09 ± 0.57 | 97.63 ± 0.81 | 0.437 | 0.153 |
| skin | CFMNN-paper | 97.77 ± 0.62 | 96.52 ± 1.00 | 95.28 ± 1.37 | 0.398 | 0.162 |
| skin | CFMNN-primary (ours) | 96.35 ± 1.10 | 94.15 ± 1.87 | 91.97 ± 2.77 | 0.305 | 0.020 |
| skin | LightGBM | 99.85 ± 0.04 | 99.77 ± 0.07 | 99.89 ± 0.05 | 0.710 | 0.008 |
| skin | LogisticRegression | 91.90 ± 0.24 | 87.98 ± 0.40 | 88.68 ± 0.54 | 0.013 | 0.000 |
| skin | RandomForest | 99.86 ± 0.04 | 99.79 ± 0.06 | 99.91 ± 0.02 | 0.366 | 0.045 |
| skin | XGBoost | 99.83 ± 0.08 | 99.75 ± 0.11 | 99.87 ± 0.07 | 0.175 | 0.003 |
| thyroid | CFMNN+Contraction | 96.88 ± 0.28 | 84.91 ± 1.24 | 80.61 ± 2.10 | 0.153 | 0.020 |
| thyroid | CFMNN-fixed | 93.89 ± 0.54 | 75.93 ± 0.82 | 79.68 ± 3.11 | 0.491 | 0.262 |
| thyroid | CFMNN-paper | 93.84 ± 0.58 | 75.72 ± 0.57 | 79.38 ± 3.03 | 0.550 | 0.304 |
| thyroid | CFMNN-primary (ours) | 96.88 ± 0.28 | 84.91 ± 1.24 | 80.61 ± 2.10 | 0.154 | 0.020 |
| thyroid | LightGBM | 99.77 ± 0.11 | 98.58 ± 0.92 | 99.26 ± 1.17 | 0.210 | 0.010 |
| thyroid | LogisticRegression | 93.91 ± 0.16 | 57.09 ± 1.63 | 53.52 ± 1.76 | 0.025 | 0.001 |
| thyroid | RandomForest | 99.70 ± 0.08 | 98.37 ± 0.81 | 99.42 ± 0.83 | 0.345 | 0.045 |
| thyroid | XGBoost | 99.56 ± 0.14 | 97.63 ± 0.82 | 98.01 ± 0.90 | 0.273 | 0.003 |

## Table C: Explanation Target Validity & Reconstruction

| Dataset   | Target                              | Target_Scope               |   Seed | Max_Reconstruction_Error    | Mean_Reconstruction_Error   | Identity_Verified   |
|:----------|:------------------------------------|:---------------------------|-------:|:----------------------------|:----------------------------|:--------------------|
| digits    | Target A (Winning-Box Membership)   | Scalar Box Membership Only |      0 | 2.22e-16                    | 5.23e-17                    | True                |
| digits    | Target B (Full Classifier Decision) | Multi-Box & Class Argmax   |      0 | N/A (Contrastive Occlusion) | N/A (Contrastive Occlusion) | True                |
| shuttle   | Target A (Winning-Box Membership)   | Scalar Box Membership Only |      0 | 2.22e-16                    | 5.80e-17                    | True                |
| shuttle   | Target B (Full Classifier Decision) | Multi-Box & Class Argmax   |      0 | N/A (Contrastive Occlusion) | N/A (Contrastive Occlusion) | True                |
| skin      | Target A (Winning-Box Membership)   | Scalar Box Membership Only |      0 | 1.11e-16                    | 3.88e-17                    | True                |
| skin      | Target B (Full Classifier Decision) | Multi-Box & Class Argmax   |      0 | N/A (Contrastive Occlusion) | N/A (Contrastive Occlusion) | True                |
| thyroid   | Target A (Winning-Box Membership)   | Scalar Box Membership Only |      0 | 3.02e-16                    | 5.62e-17                    | True                |
| thyroid   | Target B (Full Classifier Decision) | Multi-Box & Class Argmax   |      0 | N/A (Contrastive Occlusion) | N/A (Contrastive Occlusion) | True                |

## Table D: Error Detection & Central Hypothesis H2 (PD+U vs PD+U+EI_within)

| Dataset | AUROC (PD+U) | AUROC (PD+U+EI) | Δ AUROC | p-val (AUROC) | AUPRC (PD+U) | AUPRC (PD+U+EI) | Δ AUPRC | p-val (AUPRC) |
|---|---|---|---|---|---|---|---|---|
| digits | 0.978 | 0.976 | -0.002 | 0.695 | 0.683 | 0.679 | -0.004 | 0.562 |
| shuttle | 0.585 | 0.491 | -0.094 | 0.873 | 0.448 | 0.097 | -0.352 | 0.997 |
| skin | 0.963 | 0.963 | -0.000 | 0.765 | 0.439 | 0.439 | -0.000 | 0.772 |
| thyroid | 0.752 | 0.705 | -0.048 | 0.773 | 0.085 | 0.069 | -0.016 | 0.808 |

## Table E: Exact vs Approximate Caching Evaluation

| Dataset   | Cache_Mode                | Structural_Mismatch_Rate   |   Mean_Attribution_Error | Speedup   | Exactness_Proof                    |
|:----------|:--------------------------|:---------------------------|-------------------------:|:----------|:-----------------------------------|
| digits    | Approximate Cache         | 7.62%                      |                 7.79e-05 | 0.91x     | Unproven / Approximate             |
| digits    | Exact Structural Cache    | 0.00%                      |                 0        | 0.90x     | Mathematically Proven (Appendix B) |
| digits    | Full Recompute (No Cache) | 0.00%                      |                 0        | 1.00x     | Exact (Direct Evaluation)          |
| shuttle   | Approximate Cache         | 8.88%                      |                 0.00028  | 0.55x     | Unproven / Approximate             |
| shuttle   | Exact Structural Cache    | 0.00%                      |                 0        | 0.99x     | Mathematically Proven (Appendix B) |
| shuttle   | Full Recompute (No Cache) | 0.00%                      |                 0        | 1.00x     | Exact (Direct Evaluation)          |
| skin      | Approximate Cache         | 11.50%                     |                 0.0037   | 0.65x     | Unproven / Approximate             |
| skin      | Exact Structural Cache    | 0.00%                      |                 0        | 0.87x     | Mathematically Proven (Appendix B) |
| skin      | Full Recompute (No Cache) | 0.00%                      |                 0        | 1.00x     | Exact (Direct Evaluation)          |
| thyroid   | Approximate Cache         | 78.75%                     |                 0.00047  | 1.01x     | Unproven / Approximate             |
| thyroid   | Exact Structural Cache    | 0.00%                      |                 0        | 0.94x     | Mathematically Proven (Appendix B) |
| thyroid   | Full Recompute (No Cache) | 0.00%                      |                 0        | 1.00x     | Exact (Direct Evaluation)          |

## Table F: Refinement Safety & Collateral Damage Evaluation

| Dataset | Method | Test Acc (%) | Test Macro-F1 (%) | Δ Acc vs Baseline | Collateral Errors | Gate Passed Rate |
|---|---|---|---|---|---|---|
| digits | Explanation-Guided Shrink | 96.11 | 96.09 | +0.00 | 0.0 | 1.0 |
| digits | FMM Contraction | 96.11 | 96.09 | +0.00 | 0.0 | 1.0 |
| digits | No Refinement | 96.11 | 96.09 | +0.00 | 0.0 | 1.0 |
| digits | Random Shrink | 96.11 | 96.09 | +0.00 | 0.0 | 1.0 |
| shuttle | Explanation-Guided Shrink | 72.77 | 63.56 | -25.85 | 387.0 | 0.0 |
| shuttle | FMM Contraction | 30.77 | 36.80 | -67.86 | 997.3 | 0.0 |
| shuttle | No Refinement | 98.62 | 71.28 | +0.00 | 0.0 | 1.0 |
| shuttle | Random Shrink | 90.49 | 70.15 | -8.13 | 121.0 | 0.0 |
| skin | Explanation-Guided Shrink | 97.74 | 96.47 | +1.39 | 10.3 | 1.0 |
| skin | FMM Contraction | 89.57 | 80.19 | -6.78 | 126.0 | 0.0 |
| skin | No Refinement | 96.35 | 94.15 | +0.00 | 0.0 | 1.0 |
| skin | Random Shrink | 96.48 | 94.40 | +0.13 | 5.7 | 0.7 |
| thyroid | Explanation-Guided Shrink | 96.88 | 84.84 | +0.00 | 1.3 | 0.7 |
| thyroid | FMM Contraction | 96.88 | 84.91 | +0.00 | 0.0 | 1.0 |
| thyroid | No Refinement | 96.88 | 84.91 | +0.00 | 0.0 | 1.0 |
| thyroid | Random Shrink | 97.08 | 86.14 | +0.21 | 2.0 | 1.0 |

