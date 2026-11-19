# Run M: Simpson's fuzzy min-max classifier trained from scratch

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The exact check is a check; everything else is descriptive (no test). θ = 0.3, γ = 4 for both models. Means over seeds; shares pooled over seeds. Written by `scripts/run_simpson_fmm.py`.

## PACE-SHAP on SimpsonFMM against shap's ExactExplainer

**Check: PASS** (largest difference 1.4e-10; the plan's limit is 1e-09).

| Dataset | Seeds | Points | Boxes (mean) | Contractions (mean) | Largest difference | ms PACE-SHAP | ms ExactExplainer |
|---|---|---|---|---|---|---|---|
| S1 | 10 | 50 | 1,519.1 | 55.9 | 1.4e-10 | 7,765.2 | 87,691.9 |
| S2 | 10 | 50 | 1,534.3 | 173.1 | 1.6e-16 | 6,863.4 | 88,138.4 |
| S3 | 10 | 50 | 1,526.6 | 76.9 | 1.5e-16 | 5,411.5 | 68,160.5 |
| shuttle | 10 | 50 | 13.8 | 22,454.4 | 3.1e-16 | 17.9 | 375.8 |
| magic | 10 | 50 | 272.8 | 21,078.5 | 1.2e-16 | 525.2 | 15,701.3 |
| page_blocks | 10 | 50 | 25.8 | 339.8 | 1.9e-16 | 79.2 | 1,868.1 |

## SimpsonFMM beside CFMNN (13 accuracy datasets)

| Dataset | Accuracy CFMNN | Accuracy Simpson | Macro-F1 CFMNN | Macro-F1 Simpson | Boxes CFMNN | Boxes Simpson | Training error % CFMNN | Training error % Simpson | Ties % CFMNN | Ties % Simpson | Tie-rule-decided test % CFMNN | Tie-rule-decided test % Simpson | Contractions | Pairs still overlapping | Seconds CFMNN | Seconds Simpson |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 89.85 | 90.12 | 89.85 | 90.12 | 1,518.5 | 1,519.1 | 0.00 | 0.00 | nan | 0.0 | 0.0 | 0.0 | 55.9 | 0.0 | 0.5 | 1.1 |
| S2 | 81.25 | 81.23 | 81.25 | 81.22 | 1,536.8 | 1,534.3 | 0.00 | 0.01 | 100.0 | 0.0 | 0.0 | 0.0 | 173.1 | 0.0 | 0.5 | 1.1 |
| S3 | 87.84 | 87.71 | 86.47 | 86.33 | 1,527.6 | 1,526.6 | 0.00 | 0.00 | nan | nan | 0.0 | 0.0 | 76.9 | 0.0 | 0.5 | 1.0 |
| wine | 97.50 | 97.50 | 97.60 | 97.60 | 30.1 | 30.1 | 0.00 | 0.00 | nan | nan | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 | 0.0 |
| breast_cancer | 95.44 | 96.49 | 95.09 | 96.24 | 42.2 | 41.9 | 0.77 | 0.97 | 100.0 | 0.0 | 0.5 | 0.0 | 77.6 | 0.0 | 0.0 | 0.0 |
| digits | 96.06 | 95.86 | 96.04 | 95.84 | 293.3 | 293.3 | 0.00 | 0.00 | nan | nan | 0.4 | 0.4 | 0.0 | 0.0 | 0.1 | 0.3 |
| thyroid | 97.07 | 96.04 | 85.39 | 78.31 | 53.2 | 53.2 | 1.95 | 3.20 | 100.0 | 100.0 | 5.4 | 5.4 | 0.0 | 0.0 | 0.2 | 0.3 |
| shuttle | 99.77 | 85.76 | 85.88 | 75.40 | 13.4 | 13.8 | 0.19 | 14.21 | 100.0 | 0.7 | 13.0 | 0.3 | 22,454.4 | 0.0 | 1.4 | 4.2 |
| skin | 95.26 | 97.34 | 92.09 | 95.76 | 84.0 | 76.7 | 4.77 | 2.63 | 100.0 | 37.8 | 19.6 | 5.4 | 5,301.7 | 0.0 | 5.7 | 7.4 |
| magic | 70.62 | 68.68 | 69.53 | 67.00 | 280.5 | 272.8 | 27.70 | 30.61 | 100.0 | 2.1 | 77.3 | 1.5 | 21,078.5 | 0.0 | 0.7 | 3.0 |
| letter | 75.25 | 66.99 | 75.04 | 66.72 | 557.7 | 555.5 | 21.12 | 29.18 | 100.0 | 98.7 | 56.3 | 55.6 | 71.4 | 0.2 | 0.6 | 1.1 |
| landsat | 85.01 | 84.93 | 82.37 | 82.86 | 202.0 | 197.3 | 9.91 | 10.35 | 100.0 | 73.9 | 25.7 | 20.9 | 384.1 | 1.5 | 0.3 | 0.5 |
| page_blocks | 87.39 | 88.09 | 63.75 | 63.87 | 25.5 | 25.8 | 11.94 | 11.05 | 100.0 | 71.4 | 73.0 | 55.9 | 339.8 | 0.0 | 0.2 | 0.2 |

Ties: training errors inside a box of both classes (Proposition 1's form), pooled. Tie-rule-decided: test predictions whose class the tie rule decides (CFMNN: nearest centroid; Simpson: the lowest box index), pooled. Pairs still overlapping: pairs of boxes of different classes that Simpson's test still finds overlapping after training (a box made as a new point is never tested).

