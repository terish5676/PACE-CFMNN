# Run K: one large dataset (Covertype)

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Descriptive: no comparison is tested. The paper's 80/20 split, seeds 170–174 (5), θ = 0.3, γ = 4; 464,809 training rows, 116,203 test rows, 54 features, 7 classes. Mean ± standard deviation over seeds. Written by `scripts/run_covertype.py`.

## Accuracy

| Model | Test accuracy (%) | Test macro-F1 (%) | Seconds to train | Seconds to predict |
|---|---|---|---|---|
| CFMNN | 51.45 ± 2.75 | 39.11 ± 1.47 | 10.26 ± 0.23 | 4.93 ± 0.17 |
| 10-NN | 92.67 ± 0.08 | 87.11 ± 0.25 | 0.03 ± 0.00 | 34.43 ± 0.33 |
| Random Forest | 95.63 ± 0.05 | 92.69 ± 0.21 | 33.55 ± 0.52 | 0.56 ± 0.01 |

## CFMNN: size, training errors and ties

| Measure | Value |
|---|---|
| Seconds to train | 10.26 ± 0.23 |
| Boxes | 119.00 ± 3.39 |
| Training rows per box | 3,908.46 ± 109.62 |
| Training error (%) | 48.54 ± 2.73 |
| Training errors, all seeds | 1,128,010 |
| ... that are ties inside a box of both classes (Proposition 1's form) | 1,128,010 (100.00%) |
| ... whose own class reaches the winner's membership | 1,128,010 (100.00%) |
| Test predictions decided by the nearest centroid, all seeds | 574,587 of 581,015 (98.89%) |
| Test errors among those predictions, all seeds | 281,882 of 282,099 (99.92%) |

## Explanation time (50 test points per seed)

| Method | Milliseconds per explanation |
|---|---|
| PACE-flip | 1.12 ± 0.11 |
| PACE-SHAP (sampled mode, 32 orders per row, 200 rows) | 453.74 ± 24.77 |

PACE-SHAP skipped 30.55 ± 1.79% of the (row, box) pairs as unable to change the margin.

## Cost of the run

Minutes per seed: 2.20 ± 0.04. Peak memory of the process (MB): 3,057.00 ± 7.14.

