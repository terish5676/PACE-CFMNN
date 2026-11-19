# Counterfactuals: PACE vs DiCE vs nearest unlike neighbour

Paper's 80/20 split, main model (fixed centroid + search); up to 30 test points per dataset and seed; seeds [0, 1, 2]. Target = the runner-up class. Mean ± 95% CI over seeds of per-seed means. L1, features changed and plausibility are over valid counterfactuals only; lower is better for all three. Plausibility = L1 distance to the nearest training point of the target class. Time = median per counterfactual. Thyroid locks features 0 and 1 (age and sex). DiCE: 5 candidates, its own sparse output, the nearest one the model predicts as the target.

| Dataset | Method | Valid (%) | L1 distance | Features changed | Plausibility | Locked kept (%) | ms | Model calls |
|---|---|---|---|---|---|---|---|---|
| S1 | PACE | 100.0 ± 0.0 | 0.410 ± 0.064 | 1.70 ± 0.30 | 1.156 ± 0.085 | – | 2.69 ± 0.37 | 6.4 ± 0.7 |
| S1 | DiCE | 100.0 ± 0.0 | 0.416 ± 0.137 | 1.31 ± 0.19 | 1.258 ± 0.064 | – | 2500.73 ± 2.01 | – |
| S1 | Nearest unlike neighbour | 100.0 ± 0.0 | 1.432 ± 0.083 | 10.00 ± 0.00 | 0.000 ± 0.000 | – | 0.19 ± 0.01 | – |
| S2 | PACE | 100.0 ± 0.0 | 0.349 ± 0.133 | 1.46 ± 0.13 | 1.119 ± 0.068 | – | 3.22 ± 1.61 | 6.1 ± 0.9 |
| S2 | DiCE | 100.0 ± 0.0 | 0.373 ± 0.112 | 1.37 ± 0.25 | 1.194 ± 0.120 | – | 3998.42 ± 3574.42 | – |
| S2 | Nearest unlike neighbour | 100.0 ± 0.0 | 1.329 ± 0.184 | 10.00 ± 0.00 | 0.000 ± 0.000 | – | 0.21 ± 0.06 | – |
| S3 | PACE | 100.0 ± 0.0 | 0.375 ± 0.072 | 1.51 ± 0.27 | 1.121 ± 0.033 | – | 2.87 ± 0.63 | 6.1 ± 0.8 |
| S3 | DiCE | 100.0 ± 0.0 | 0.416 ± 0.033 | 1.29 ± 0.10 | 1.230 ± 0.061 | – | 2857.96 ± 817.24 | – |
| S3 | Nearest unlike neighbour | 100.0 ± 0.0 | 1.390 ± 0.119 | 10.00 ± 0.00 | 0.000 ± 0.000 | – | 0.16 ± 0.04 | – |
| wine | PACE | 100.0 ± 0.0 | 0.695 ± 0.110 | 3.44 ± 0.42 | 1.367 ± 0.129 | – | 1.58 ± 0.03 | 8.0 ± 1.2 |
| wine | DiCE | 100.0 ± 0.0 | 0.808 ± 0.146 | 2.23 ± 0.52 | 1.617 ± 0.081 | – | 248.05 ± 37.21 | – |
| wine | Nearest unlike neighbour | 100.0 ± 0.0 | 1.954 ± 0.172 | 12.69 ± 0.05 | 0.000 ± 0.000 | – | 0.03 ± 0.00 | – |
| breast_cancer | PACE | 100.0 ± 0.0 | 1.216 ± 0.609 | 9.68 ± 4.84 | 1.747 ± 0.100 | – | 4.33 ± 1.66 | 23.1 ± 2.4 |
| breast_cancer | DiCE | 100.0 ± 0.0 | 1.560 ± 0.968 | 4.21 ± 2.40 | 2.884 ± 0.626 | – | 486.57 ± 258.68 | – |
| breast_cancer | Nearest unlike neighbour | 100.0 ± 0.0 | 2.545 ± 0.501 | 29.98 ± 0.10 | 0.153 ± 0.365 | – | 0.05 ± 0.00 | – |
| thyroid | PACE | 98.9 ± 4.8 | 0.169 ± 0.135 | 1.81 ± 0.35 | 0.238 ± 0.033 | 100.0 ± 0.0 | 1.63 ± 0.70 | 6.9 ± 1.9 |
| thyroid | DiCE | 98.9 ± 4.8 | 0.298 ± 0.068 | 1.48 ± 0.40 | 0.538 ± 0.325 | 100.0 ± 0.0 | 383.47 ± 9.57 | – |
| thyroid | Nearest unlike neighbour | 100.0 ± 0.0 | 0.497 ± 0.335 | 5.98 ± 0.17 | 0.007 ± 0.018 | 5.6 ± 4.8 | 0.14 ± 0.01 | – |
| shuttle | PACE | 100.0 ± 0.0 | 0.101 ± 0.031 | 3.27 ± 0.84 | 0.150 ± 0.026 | – | 2.68 ± 0.25 | 9.3 ± 1.3 |
| shuttle | DiCE | 100.0 ± 0.0 | 0.239 ± 0.061 | 1.31 ± 0.31 | 0.291 ± 0.077 | – | 437.81 ± 74.93 | – |
| shuttle | Nearest unlike neighbour | 100.0 ± 0.0 | 0.230 ± 0.037 | 7.43 ± 0.60 | 0.012 ± 0.009 | – | 0.43 ± 0.02 | – |
| magic | PACE | 100.0 ± 0.0 | 0.223 ± 0.168 | 2.26 ± 1.43 | 0.247 ± 0.049 | – | 2.83 ± 0.19 | 12.6 ± 1.7 |
| magic | DiCE | 100.0 ± 0.0 | 0.297 ± 0.040 | 1.47 ± 0.22 | 0.433 ± 0.062 | – | 1213.85 ± 346.00 | – |
| magic | Nearest unlike neighbour | 100.0 ± 0.0 | 0.291 ± 0.042 | 9.99 ± 0.05 | 0.160 ± 0.126 | – | 0.90 ± 0.32 | – |
| letter | PACE | 100.0 ± 0.0 | 0.206 ± 0.073 | 2.74 ± 0.54 | 0.732 ± 0.081 | – | 12.22 ± 1.46 | 16.1 ± 1.3 |
| letter | DiCE | 98.9 ± 4.8 | 0.345 ± 0.049 | 1.47 ± 0.02 | 0.966 ± 0.088 | – | 5084.25 ± 4279.44 | – |
| letter | Nearest unlike neighbour | 100.0 ± 0.0 | 0.745 ± 0.090 | 7.63 ± 0.58 | 0.305 ± 0.067 | – | 0.52 ± 0.17 | – |
| landsat | PACE | 100.0 ± 0.0 | 1.166 ± 0.395 | 11.23 ± 0.38 | 1.633 ± 0.204 | – | 11.53 ± 0.33 | 33.5 ± 3.0 |
| landsat | DiCE | 98.9 ± 4.8 | 1.171 ± 0.751 | 3.26 ± 1.34 | 2.551 ± 0.497 | – | 1435.18 ± 481.08 | – |
| landsat | Nearest unlike neighbour | 100.0 ± 0.0 | 2.302 ± 0.672 | 30.27 ± 0.33 | 0.582 ± 0.446 | – | 0.18 ± 0.04 | – |
| page_blocks | PACE | 100.0 ± 0.0 | 0.291 ± 0.071 | 1.80 ± 0.46 | 0.205 ± 0.027 | – | 2.90 ± 0.12 | 12.9 ± 0.9 |
| page_blocks | DiCE | 100.0 ± 0.0 | 0.337 ± 0.282 | 1.38 ± 0.25 | 0.513 ± 0.159 | – | 231.30 ± 21.85 | – |
| page_blocks | Nearest unlike neighbour | 100.0 ± 0.0 | 0.288 ± 0.123 | 9.54 ± 0.66 | 0.161 ± 0.035 | – | 0.07 ± 0.02 | – |

## PACE minus DiCE, per point

Points where both counterfactuals are valid, seeds pooled; two-sided Wilcoxon signed-rank test, Holm-corrected across datasets for each measure. Negative = PACE lower (better for all three).

| Dataset | Measure | Difference | Points where PACE is lower | p | p (Holm) |
|---|---|---|---|---|---|
| S1 | L1 distance | -0.006 | 46 of 90 | 0.577 | 1 |
| S1 | Features changed | +0.389 | 8 of 90 | 3.88e-05 | 0.000194 |
| S1 | Plausibility | -0.102 | 58 of 90 | 0.00237 | 0.00237 |
| S2 | L1 distance | -0.024 | 44 of 90 | 0.293 | 0.878 |
| S2 | Features changed | +0.089 | 20 of 90 | 0.273 | 0.273 |
| S2 | Plausibility | -0.075 | 62 of 90 | 0.000594 | 0.00119 |
| S3 | L1 distance | -0.041 | 52 of 90 | 0.04 | 0.2 |
| S3 | Features changed | +0.222 | 15 of 90 | 0.0181 | 0.0363 |
| S3 | Plausibility | -0.109 | 62 of 90 | 1.65e-05 | 4.96e-05 |
| wine | L1 distance | -0.113 | 66 of 90 | 5.29e-06 | 4.23e-05 |
| wine | Features changed | +1.211 | 5 of 90 | 9.53e-11 | 7.62e-10 |
| wine | Plausibility | -0.250 | 78 of 90 | 2.89e-12 | 1.15e-11 |
| breast_cancer | L1 distance | -0.344 | 63 of 90 | 1.43e-05 | 9.99e-05 |
| breast_cancer | Features changed | +5.467 | 2 of 90 | 1.02e-15 | 1.12e-14 |
| breast_cancer | Plausibility | -1.137 | 90 of 90 | 1.74e-16 | 1.92e-15 |
| thyroid | L1 distance | -0.128 | 76 of 88 | 4.24e-09 | 4.24e-08 |
| thyroid | Features changed | +0.330 | 11 of 88 | 0.000896 | 0.00269 |
| thyroid | Plausibility | -0.298 | 77 of 88 | 1.57e-14 | 1.1e-13 |
| shuttle | L1 distance | -0.138 | 77 of 90 | 3.18e-14 | 3.5e-13 |
| shuttle | Features changed | +1.956 | 4 of 90 | 1.8e-13 | 1.62e-12 |
| shuttle | Plausibility | -0.141 | 79 of 90 | 5.11e-15 | 4.09e-14 |
| magic | L1 distance | -0.074 | 57 of 90 | 0.00224 | 0.0135 |
| magic | Features changed | +0.789 | 14 of 90 | 7.62e-07 | 4.57e-06 |
| magic | Plausibility | -0.186 | 80 of 90 | 1.66e-13 | 9.59e-13 |
| letter | L1 distance | -0.143 | 70 of 89 | 1.35e-07 | 1.22e-06 |
| letter | Features changed | +1.247 | 10 of 89 | 7.37e-08 | 5.16e-07 |
| letter | Plausibility | -0.235 | 77 of 89 | 1.6e-13 | 9.59e-13 |
| landsat | L1 distance | +0.005 | 41 of 89 | 0.178 | 0.711 |
| landsat | Features changed | +7.989 | 1 of 89 | 1.88e-15 | 1.88e-14 |
| landsat | Plausibility | -0.916 | 87 of 89 | 3.13e-16 | 2.81e-15 |
| page_blocks | L1 distance | -0.046 | 42 of 90 | 0.986 | 1 |
| page_blocks | Features changed | +0.422 | 15 of 90 | 0.000228 | 0.000912 |
| page_blocks | Plausibility | -0.308 | 88 of 90 | 1.99e-16 | 1.99e-15 |
