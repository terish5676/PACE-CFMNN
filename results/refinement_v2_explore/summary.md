# Refinement v2: exploratory checks E1–E4

Seeds 0, 1, 2; descriptive only (means over seeds, no p-values). v1's split, budget (min(300, a quarter of the pool)), 10 rounds, gate and box cap. Written by `scripts/run_refinement_explore.py` from `runs.csv`.

## Labels used beyond the training rows

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B9 refit, all pool labels | 1,440 | 1,287 | 1,094 | 3,804 | 4,000 | 49,011 |
| OE | A+B | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| OE | A+B+C'-volume | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| OE | A+B+C | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| OE | A+B+C-fewest | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| OE | A+B+C-paceshap | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| OE | A+B+C-oracle | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| random | A+B+C'-volume | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| margin | A+B+C'-volume | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| knn | A+B+C'-volume | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| oracle_errors | A+B+C'-volume | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| kNN, train only | 0 | 0 | 0 | 0 | 0 | 0 |
| RF, train only | 0 | 0 | 0 | 0 | 0 | 0 |
| CFMNN refit, +bought | 300 | 300 | 273 | 300 | 300 | 300 |
| kNN, +bought | 300 | 300 | 273 | 300 | 300 | 300 |
| RF, +bought | 300 | 300 | 273 | 300 | 300 | 300 |
| CFMNN refit, +bought +acceptance | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| kNN, +bought +acceptance | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| RF, +bought +acceptance | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| SR-300 | 720 | 644 | 548 | 1,902 | 2,000 | 24,506 |
| SR-300-train | 0 | 0 | 0 | 0 | 0 | 0 |
| SR-all | 720 | 644 | 548 | 1,902 | 2,000 | 24,506 |
| SR-all + OE | A+B+C'-volume | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |

## Test macro-F1 (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 84.7 | 82.7 | 76.9 | 73.3 | 74.3 | 90.1 |
| B9 refit, all pool labels | 85.1 | 82.6 | 70.7 | 73.9 | 75.5 | 90.6 |
| OE | A+B | 86.8 | 82.3 | 75.1 | 73.5 | 75.1 | 90.5 |
| OE | A+B+C'-volume | 90.8 | 83.2 | 78.1 | 78.3 | 76.9 | 99.0 |
| OE | A+B+C | 87.9 | 83.1 | 79.3 | 76.8 | 76.7 | 98.8 |
| OE | A+B+C-fewest | 89.2 | 83.3 | 81.6 | 78.1 | 77.0 | 99.0 |
| OE | A+B+C-paceshap | 90.4 | 82.7 | 80.0 | 77.7 | 76.7 | 98.5 |
| OE | A+B+C-oracle | 90.3 | 84.0 | 79.2 | 79.2 | 78.0 | 99.4 |
| random | A+B+C'-volume | 87.2 | 83.0 | 78.5 | 75.2 | 75.4 | 97.3 |
| margin | A+B+C'-volume | 89.9 | 83.3 | 77.9 | 75.1 | 76.0 | 98.2 |
| knn | A+B+C'-volume | 89.5 | 83.3 | 77.2 | 78.9 | 77.4 | 99.5 |
| oracle_errors | A+B+C'-volume | 91.9 | 83.7 | 78.3 | 75.3 | 77.0 | 99.2 |
| kNN, train only | 54.5 | 86.5 | 59.6 | 79.4 | 91.7 | 99.9 |
| RF, train only | 97.2 | 88.9 | 84.6 | 85.4 | 95.0 | 99.9 |
| CFMNN refit, +bought | 85.4 | 82.6 | 73.6 | 73.5 | 74.6 | 90.2 |
| kNN, +bought | 57.8 | 87.0 | 67.0 | 79.4 | 92.1 | 99.9 |
| RF, +bought | 97.2 | 89.5 | 84.7 | 85.5 | 95.3 | 99.9 |
| CFMNN refit, +bought +acceptance | 84.1 | 82.4 | 72.6 | 73.9 | 74.9 | 89.0 |
| kNN, +bought +acceptance | 59.5 | 86.8 | 69.9 | 79.8 | 92.7 | 99.9 |
| RF, +bought +acceptance | 96.8 | 89.7 | 84.1 | 85.9 | 96.1 | 99.9 |
| SR-300 | 90.5 | 83.5 | 77.1 | 75.5 | 76.4 | 98.4 |
| SR-300-train | 91.0 | 83.8 | 77.7 | 75.9 | 76.8 | 98.6 |
| SR-all | 90.5 | 83.5 | 77.1 | 72.5 | 77.7 | 98.3 |
| SR-all + OE | A+B+C'-volume | 91.5 | 83.0 | 77.8 | 77.8 | 78.8 | 99.5 |

## Change in test macro-F1 against B0 (points)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B9 refit, all pool labels | +0.38 | -0.09 | -6.24 | +0.59 | +1.19 | +0.51 |
| OE | A+B | +2.11 | -0.37 | -1.77 | +0.22 | +0.78 | +0.43 |
| OE | A+B+C'-volume | +6.11 | +0.57 | +1.24 | +4.99 | +2.57 | +8.90 |
| OE | A+B+C | +3.17 | +0.47 | +2.42 | +3.50 | +2.42 | +8.68 |
| OE | A+B+C-fewest | +4.48 | +0.68 | +4.66 | +4.75 | +2.72 | +8.88 |
| OE | A+B+C-paceshap | +5.67 | +0.03 | +3.09 | +4.41 | +2.38 | +8.38 |
| OE | A+B+C-oracle | +5.54 | +1.31 | +2.33 | +5.83 | +3.70 | +9.28 |
| random | A+B+C'-volume | +2.47 | +0.38 | +1.63 | +1.92 | +1.15 | +7.17 |
| margin | A+B+C'-volume | +5.14 | +0.58 | +0.96 | +1.75 | +1.69 | +8.09 |
| knn | A+B+C'-volume | +4.73 | +0.60 | +0.29 | +5.58 | +3.14 | +9.37 |
| oracle_errors | A+B+C'-volume | +7.13 | +1.08 | +1.36 | +2.00 | +2.72 | +9.14 |
| kNN, train only | -30.19 | +3.79 | -17.33 | +6.05 | +17.46 | +9.82 |
| RF, train only | +12.50 | +6.20 | +7.68 | +12.05 | +20.68 | +9.85 |
| CFMNN refit, +bought | +0.68 | -0.10 | -3.30 | +0.18 | +0.27 | +0.16 |
| kNN, +bought | -26.92 | +4.28 | -9.86 | +6.07 | +17.78 | +9.82 |
| RF, +bought | +12.46 | +6.79 | +7.80 | +12.14 | +21.03 | +9.85 |
| CFMNN refit, +bought +acceptance | -0.62 | -0.28 | -4.33 | +0.57 | +0.66 | -1.13 |
| kNN, +bought +acceptance | -25.24 | +4.17 | -6.95 | +6.50 | +18.44 | +9.84 |
| RF, +bought +acceptance | +12.07 | +7.01 | +7.18 | +12.52 | +21.79 | +9.85 |
| SR-300 | +5.80 | +0.85 | +0.17 | +2.20 | +2.15 | +8.31 |
| SR-300-train | +6.27 | +1.16 | +0.80 | +2.55 | +2.50 | +8.51 |
| SR-all | +5.80 | +0.79 | +0.17 | -0.83 | +3.42 | +8.25 |
| SR-all + OE | A+B+C'-volume | +6.75 | +0.30 | +0.91 | +4.48 | +4.53 | +9.38 |

## Lowest class recall (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 61.1 | 58.7 | 58.9 | 71.7 | 44.1 | 73.9 |
| B9 refit, all pool labels | 59.0 | 56.1 | 60.1 | 72.7 | 47.9 | 75.5 |
| OE | A+B | 63.9 | 55.0 | 58.9 | 72.2 | 45.7 | 75.0 |
| OE | A+B+C'-volume | 76.9 | 55.6 | 51.5 | 73.4 | 48.3 | 97.9 |
| OE | A+B+C | 74.6 | 58.2 | 59.8 | 70.5 | 52.7 | 97.5 |
| OE | A+B+C-fewest | 69.4 | 55.6 | 57.9 | 73.0 | 53.6 | 97.7 |
| OE | A+B+C-paceshap | 76.9 | 54.0 | 60.1 | 69.3 | 46.4 | 96.8 |
| OE | A+B+C-oracle | 75.6 | 56.1 | 44.2 | 73.3 | 53.1 | 98.7 |
| random | A+B+C'-volume | 68.5 | 58.2 | 54.3 | 70.6 | 44.6 | 93.4 |
| margin | A+B+C'-volume | 76.9 | 57.7 | 44.2 | 69.8 | 50.5 | 95.3 |
| knn | A+B+C'-volume | 72.2 | 60.3 | 54.3 | 66.1 | 53.2 | 99.2 |
| oracle_errors | A+B+C'-volume | 80.6 | 61.9 | 54.3 | 69.6 | 51.8 | 98.4 |
| kNN, train only | 2.7 | 61.9 | 6.1 | 59.9 | 80.5 | 99.9 |
| RF, train only | 96.9 | 64.6 | 52.5 | 75.1 | 85.8 | 99.9 |
| CFMNN refit, +bought | 60.9 | 53.4 | 62.6 | 71.8 | 43.7 | 74.4 |
| kNN, +bought | 6.4 | 66.1 | 29.0 | 59.9 | 80.1 | 99.9 |
| RF, +bought | 95.0 | 67.7 | 55.6 | 75.1 | 87.8 | 99.9 |
| CFMNN refit, +bought +acceptance | 58.1 | 54.0 | 62.6 | 72.5 | 46.6 | 71.9 |
| kNN, +bought +acceptance | 9.1 | 64.6 | 32.1 | 60.2 | 82.1 | 99.9 |
| RF, +bought +acceptance | 92.9 | 66.1 | 55.6 | 75.5 | 88.5 | 99.9 |
| SR-300 | 78.5 | 58.2 | 45.5 | 67.8 | 50.3 | 96.2 |
| SR-300-train | 80.6 | 60.3 | 64.6 | 68.3 | 50.7 | 96.7 |
| SR-all | 78.5 | 59.3 | 45.5 | 69.4 | 56.1 | 95.4 |
| SR-all + OE | A+B+C'-volume | 80.6 | 58.2 | 48.2 | 70.0 | 55.7 | 99.1 |

## E1: self-repair

### Training error at the end (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| SR-300 | 0.3 | 7.3 | 5.5 | 19.6 | 18.6 | 1.0 |
| SR-300-train | 0.3 | 6.4 | 6.2 | 19.8 | 17.5 | 0.9 |
| SR-all | 0.3 | 7.2 | 5.5 | 23.6 | 15.5 | 1.1 |
| SR-all + OE | A+B+C'-volume | 0.5 | 7.6 | 5.2 | 18.5 | 15.2 | 0.4 |
| OE | A+B+C'-volume | 0.7 | 8.6 | 5.3 | 18.9 | 19.0 | 0.7 |
| B0 | 1.6 | 9.2 | 6.9 | 22.9 | 21.2 | 5.8 |

### Accepted shrinks

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| SR-300 | 14 | 150 | 18 | 203 | 152 | 36 |
| SR-300-train | 14 | 162 | 11 | 206 | 194 | 36 |
| SR-all | 14 | 156 | 18 | 782 | 642 | 74 |

## E2: which feature to cut

### Accepted shrinks

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B+C | 15 | 61 | 12 | 157 | 106 | 55 |
| OE | A+B+C'-volume | 14 | 73 | 12 | 171 | 125 | 70 |
| OE | A+B+C-fewest | 23 | 80 | 25 | 171 | 155 | 69 |
| OE | A+B+C-paceshap | 15 | 52 | 16 | 158 | 104 | 62 |
| OE | A+B+C-oracle | 15 | 85 | 24 | 153 | 178 | 54 |

### Mean share of the box's width cut

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B+C | 0.218 | 0.278 | 0.160 | 0.270 | 0.216 | 0.192 |
| OE | A+B+C'-volume | 0.001 | 0.043 | 0.011 | 0.067 | 0.005 | 0.093 |
| OE | A+B+C-fewest | 0.128 | 0.115 | 0.124 | 0.119 | 0.045 | 0.144 |
| OE | A+B+C-paceshap | 0.331 | 0.270 | 0.210 | 0.262 | 0.196 | 0.190 |
| OE | A+B+C-oracle | 0.075 | 0.157 | 0.092 | 0.209 | 0.138 | 0.150 |

### Mean training members cut

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B+C | 31 | 5 | 29 | 21 | 8 | 102 |
| OE | A+B+C'-volume | 125 | 3 | 12 | 7 | 4 | 85 |
| OE | A+B+C-fewest | 3 | 2 | 5 | 4 | 2 | 64 |
| OE | A+B+C-paceshap | 86 | 7 | 26 | 19 | 7 | 91 |
| OE | A+B+C-oracle | 116 | 5 | 6 | 22 | 8 | 102 |

## E3: which points to buy

### Share of bought points the model got wrong (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| random | A+B+C'-volume | 2.8 | 14.1 | 7.4 | 23.2 | 25.9 | 2.7 |
| margin | A+B+C'-volume | 5.0 | 33.3 | 12.8 | 27.3 | 41.4 | 8.7 |
| OE | A+B+C'-volume | 10.3 | 47.6 | 22.1 | 65.2 | 88.3 | 35.1 |
| knn | A+B+C'-volume | 7.4 | 54.6 | 25.5 | 69.1 | 98.7 | 57.8 |
| oracle_errors | A+B+C'-volume | 11.0 | 68.7 | 29.7 | 94.8 | 98.6 | 62.4 |

## E4: auditability

Audit rows: a fixed sample of at most 2,000 test rows. Churn is against B0.

### Predictions changed (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B9 refit, all pool labels | 0.79 | 4.30 | 4.20 | 7.27 | 10.57 | 1.07 |
| OE | A+B | 0.51 | 3.31 | 1.64 | 3.03 | 2.80 | 0.28 |
| OE | A+B+C'-volume | 1.57 | 6.06 | 3.89 | 14.49 | 9.10 | 5.43 |
| OE | A+B+C | 1.53 | 7.51 | 3.47 | 20.24 | 9.95 | 5.53 |
| OE | A+B+C-fewest | 0.97 | 4.97 | 5.29 | 12.60 | 7.70 | 5.20 |
| OE | A+B+C-paceshap | 1.48 | 7.61 | 3.83 | 19.09 | 9.40 | 5.38 |
| OE | A+B+C-oracle | 1.67 | 7.66 | 6.57 | 18.26 | 14.47 | 5.50 |
| random | A+B+C'-volume | 0.74 | 3.57 | 2.19 | 10.52 | 4.48 | 4.07 |
| margin | A+B+C'-volume | 1.34 | 4.09 | 5.05 | 12.78 | 6.43 | 5.33 |
| knn | A+B+C'-volume | 0.93 | 5.38 | 4.26 | 16.61 | 9.12 | 5.57 |
| oracle_errors | A+B+C'-volume | 1.57 | 5.54 | 4.01 | 18.58 | 10.07 | 5.50 |
| kNN, train only | 5.09 | 14.08 | 6.87 | 24.33 | 25.77 | 5.73 |
| RF, train only | 3.29 | 11.85 | 6.45 | 23.20 | 25.08 | 5.73 |
| CFMNN refit, +bought | 0.46 | 3.73 | 2.98 | 2.35 | 3.32 | 0.58 |
| kNN, +bought | 5.09 | 13.61 | 6.87 | 24.55 | 25.87 | 5.73 |
| RF, +bought | 3.15 | 11.96 | 6.33 | 23.05 | 25.00 | 5.73 |
| CFMNN refit, +bought +acceptance | 0.60 | 4.87 | 4.87 | 6.06 | 7.93 | 1.35 |
| kNN, +bought +acceptance | 4.81 | 13.46 | 6.69 | 24.38 | 25.68 | 5.70 |
| RF, +bought +acceptance | 3.10 | 12.27 | 6.45 | 23.08 | 25.12 | 5.70 |
| SR-300 | 1.48 | 7.14 | 5.66 | 19.84 | 8.67 | 5.65 |
| SR-300-train | 1.44 | 6.63 | 2.92 | 19.70 | 10.18 | 5.63 |
| SR-all | 1.48 | 7.40 | 5.66 | 29.00 | 16.58 | 6.00 |
| SR-all + OE | A+B+C'-volume | 1.71 | 6.78 | 5.41 | 23.62 | 16.93 | 5.62 |

### Predictions fixed (wrong → right)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B9 refit, all pool labels | 4 | 13 | 4 | 74 | 84 | 13 |
| OE | A+B | 3 | 10 | 3 | 31 | 26 | 5 |
| OE | A+B+C'-volume | 10 | 21 | 14 | 188 | 91 | 107 |
| OE | A+B+C | 8 | 25 | 14 | 230 | 89 | 106 |
| OE | A+B+C-fewest | 7 | 18 | 20 | 168 | 77 | 102 |
| OE | A+B+C-paceshap | 9 | 24 | 15 | 230 | 86 | 102 |
| OE | A+B+C-oracle | 10 | 27 | 24 | 232 | 137 | 109 |
| random | A+B+C'-volume | 4 | 12 | 9 | 122 | 41 | 79 |
| margin | A+B+C'-volume | 9 | 15 | 18 | 142 | 62 | 99 |
| knn | A+B+C'-volume | 6 | 18 | 15 | 221 | 96 | 110 |
| oracle_errors | A+B+C'-volume | 11 | 20 | 15 | 200 | 100 | 108 |
| kNN, train only | 7 | 55 | 22 | 307 | 392 | 114 |
| RF, train only | 21 | 56 | 28 | 337 | 431 | 114 |
| CFMNN refit, +bought | 2 | 12 | 2 | 24 | 25 | 8 |
| kNN, +bought | 7 | 54 | 23 | 309 | 397 | 114 |
| RF, +bought | 21 | 58 | 28 | 337 | 433 | 114 |
| CFMNN refit, +bought +acceptance | 2 | 15 | 3 | 62 | 60 | 10 |
| kNN, +bought +acceptance | 7 | 54 | 24 | 311 | 404 | 114 |
| RF, +bought +acceptance | 21 | 60 | 28 | 340 | 447 | 114 |
| SR-300 | 10 | 24 | 16 | 216 | 76 | 103 |
| SR-300-train | 10 | 24 | 6 | 218 | 88 | 104 |
| SR-all | 10 | 24 | 16 | 271 | 150 | 106 |
| SR-all + OE | A+B+C'-volume | 11 | 22 | 18 | 273 | 164 | 110 |

### Predictions broken (right → wrong)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B9 refit, all pool labels | 2 | 12 | 17 | 65 | 60 | 8 |
| OE | A+B | 0 | 10 | 6 | 27 | 11 | 1 |
| OE | A+B+C'-volume | 1 | 16 | 6 | 88 | 40 | 2 |
| OE | A+B+C | 3 | 21 | 5 | 155 | 42 | 4 |
| OE | A+B+C-fewest | 0 | 11 | 7 | 72 | 24 | 2 |
| OE | A+B+C-paceshap | 1 | 23 | 6 | 133 | 39 | 6 |
| OE | A+B+C-oracle | 2 | 18 | 11 | 115 | 66 | 1 |
| random | A+B+C'-volume | 1 | 9 | 3 | 78 | 18 | 3 |
| margin | A+B+C'-volume | 1 | 11 | 8 | 101 | 28 | 8 |
| knn | A+B+C'-volume | 0 | 15 | 7 | 95 | 35 | 2 |
| oracle_errors | A+B+C'-volume | 1 | 14 | 7 | 153 | 47 | 2 |
| kNN, train only | 29 | 31 | 15 | 156 | 48 | 1 |
| RF, train only | 2 | 18 | 7 | 104 | 23 | 1 |
| CFMNN refit, +bought | 1 | 10 | 14 | 21 | 19 | 4 |
| kNN, +bought | 29 | 28 | 13 | 158 | 46 | 1 |
| RF, +bought | 2 | 16 | 6 | 102 | 18 | 1 |
| CFMNN refit, +bought +acceptance | 2 | 14 | 23 | 53 | 46 | 17 |
| kNN, +bought +acceptance | 27 | 28 | 12 | 152 | 41 | 0 |
| RF, +bought +acceptance | 2 | 16 | 7 | 99 | 16 | 0 |
| SR-300 | 1 | 19 | 12 | 161 | 34 | 10 |
| SR-300-train | 1 | 17 | 7 | 157 | 39 | 9 |
| SR-all | 1 | 20 | 12 | 281 | 84 | 14 |
| SR-all + OE | A+B+C'-volume | 1 | 19 | 10 | 177 | 75 | 2 |

### E4, tight certificate

### Changes outside it (must be 0)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 0 | 0 | 0 | 0 | 0 | 0 |
| OE | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |
| OE | A+B+C | 0 | 0 | 0 | 0 | 0 | 0 |
| OE | A+B+C-fewest | 0 | 0 | 0 | 0 | 0 | 0 |
| OE | A+B+C-paceshap | 0 | 0 | 0 | 0 | 0 | 0 |
| OE | A+B+C-oracle | 0 | 0 | 0 | 0 | 0 | 0 |
| random | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |
| margin | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |
| knn | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |
| oracle_errors | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |
| SR-300 | 0 | 0 | 0 | 0 | 0 | 0 |
| SR-300-train | 0 | 0 | 0 | 0 | 0 | 0 |
| SR-all | 0 | 0 | 0 | 0 | 0 | 0 |
| SR-all + OE | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |

### Size per edit (% of audit rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 1.61 | 3.08 | 35.95 | 14.83 | 1.49 | 7.86 |
| OE | A+B+C'-volume | 1.31 | 3.07 | 30.71 | 15.08 | 1.96 | 6.66 |
| OE | A+B+C | 1.57 | 3.04 | 29.06 | 12.48 | 1.99 | 6.31 |
| OE | A+B+C-fewest | 1.17 | 3.12 | 21.78 | 15.19 | 2.09 | 5.50 |
| OE | A+B+C-paceshap | 1.20 | 2.69 | 26.94 | 11.98 | 1.82 | 5.65 |
| OE | A+B+C-oracle | 1.23 | 2.66 | 24.24 | 11.02 | 2.16 | 5.54 |
| random | A+B+C'-volume | 0.90 | 2.07 | 31.41 | 14.25 | 1.36 | 2.07 |
| margin | A+B+C'-volume | 1.67 | 4.03 | 26.43 | 17.09 | 1.86 | 8.75 |
| knn | A+B+C'-volume | 1.39 | 2.72 | 23.44 | 14.66 | 2.09 | 6.46 |
| oracle_errors | A+B+C'-volume | 1.21 | 2.64 | 25.81 | 14.27 | 1.91 | 6.20 |
| SR-300 | 1.07 | 2.19 | 9.61 | 7.77 | 1.37 | 3.49 |
| SR-300-train | 1.06 | 2.39 | 10.38 | 7.79 | 1.44 | 3.48 |
| SR-all | 1.07 | 2.17 | 9.61 | 4.10 | 1.11 | 3.11 |
| SR-all + OE | A+B+C'-volume | 1.28 | 2.01 | 16.63 | 3.61 | 1.12 | 3.35 |

### Precision: changed rows / certified rows, over all edits (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 0.2 | 0.7 | 0.0 | 0.1 | 1.0 | 0.0 |
| OE | A+B+C'-volume | 0.7 | 1.6 | 0.1 | 0.8 | 2.6 | 0.4 |
| OE | A+B+C | 0.6 | 2.3 | 0.1 | 1.8 | 2.9 | 0.5 |
| OE | A+B+C-fewest | 0.5 | 1.3 | 0.2 | 0.6 | 2.1 | 0.5 |
| OE | A+B+C-paceshap | 0.9 | 2.4 | 0.2 | 1.7 | 2.8 | 0.6 |
| OE | A+B+C-oracle | 0.8 | 3.3 | 0.2 | 1.4 | 4.4 | 0.4 |
| random | A+B+C'-volume | 0.6 | 1.1 | 0.1 | 0.5 | 1.5 | 0.7 |
| margin | A+B+C'-volume | 0.4 | 1.0 | 0.1 | 0.5 | 1.6 | 0.3 |
| knn | A+B+C'-volume | 0.5 | 1.7 | 0.2 | 0.9 | 2.6 | 0.6 |
| oracle_errors | A+B+C'-volume | 0.7 | 2.4 | 0.2 | 1.5 | 2.9 | 0.7 |
| SR-300 | 9.4 | 4.6 | 5.8 | 2.8 | 5.5 | 6.3 |
| SR-300-train | 9.0 | 4.3 | 3.7 | 2.7 | 5.4 | 6.3 |
| SR-all | 9.4 | 4.7 | 5.8 | 3.7 | 5.0 | 3.6 |
| SR-all + OE | A+B+C'-volume | 0.8 | 3.2 | 0.3 | 4.0 | 4.4 | 1.0 |

### Rows some edit could have changed (% of audit rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 4.9 | 22.9 | 59.5 | 72.6 | 46.9 | 21.9 |
| OE | A+B+C'-volume | 5.0 | 23.4 | 58.0 | 73.9 | 49.8 | 22.0 |
| OE | A+B+C | 5.2 | 23.7 | 55.9 | 74.7 | 49.8 | 22.1 |
| OE | A+B+C-fewest | 5.0 | 23.1 | 53.6 | 73.7 | 50.2 | 22.0 |
| OE | A+B+C-paceshap | 5.5 | 23.8 | 51.3 | 74.5 | 49.6 | 22.1 |
| OE | A+B+C-oracle | 5.0 | 23.7 | 57.5 | 74.6 | 50.3 | 22.0 |
| random | A+B+C'-volume | 4.7 | 22.1 | 46.0 | 72.5 | 48.1 | 21.9 |
| margin | A+B+C'-volume | 4.6 | 21.9 | 41.9 | 72.0 | 49.1 | 21.9 |
| knn | A+B+C'-volume | 4.8 | 23.1 | 50.1 | 73.3 | 50.0 | 22.1 |
| oracle_errors | A+B+C'-volume | 4.9 | 24.0 | 49.6 | 74.2 | 50.2 | 22.1 |
| SR-300 | 4.4 | 21.9 | 41.1 | 73.0 | 45.1 | 21.6 |
| SR-300-train | 4.4 | 21.8 | 33.2 | 73.0 | 47.0 | 21.6 |
| SR-all | 4.4 | 22.1 | 41.1 | 75.2 | 50.7 | 21.7 |
| SR-all + OE | A+B+C'-volume | 5.0 | 24.0 | 44.7 | 76.1 | 51.4 | 22.0 |

### Explained rows outside it (of 100)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 94 | 76 | 41 | 33 | 52 | 76 |
| OE | A+B+C'-volume | 94 | 75 | 45 | 32 | 51 | 76 |
| OE | A+B+C | 94 | 75 | 47 | 30 | 49 | 76 |
| OE | A+B+C-fewest | 94 | 75 | 52 | 32 | 49 | 76 |
| OE | A+B+C-paceshap | 94 | 75 | 52 | 30 | 49 | 76 |
| OE | A+B+C-oracle | 94 | 76 | 45 | 29 | 50 | 76 |
| random | A+B+C'-volume | 94 | 76 | 58 | 32 | 54 | 76 |
| margin | A+B+C'-volume | 94 | 76 | 63 | 34 | 51 | 76 |
| knn | A+B+C'-volume | 94 | 76 | 54 | 33 | 51 | 76 |
| oracle_errors | A+B+C'-volume | 94 | 74 | 56 | 31 | 52 | 76 |
| SR-300 | 94 | 77 | 64 | 32 | 55 | 76 |
| SR-300-train | 94 | 77 | 71 | 32 | 52 | 76 |
| SR-all | 94 | 77 | 64 | 29 | 51 | 76 |
| SR-all + OE | A+B+C'-volume | 94 | 76 | 60 | 27 | 49 | 76 |

### PACE-flip top-3 changed on those rows (share)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 0.10 | 0.29 | 0.28 | 0.11 | 0.09 | 0.00 |
| OE | A+B+C'-volume | 0.19 | 0.35 | 0.32 | 0.39 | 0.22 | 0.00 |
| OE | A+B+C | 0.28 | 0.43 | 0.55 | 0.55 | 0.28 | 0.00 |
| OE | A+B+C-fewest | 0.17 | 0.34 | 0.57 | 0.38 | 0.14 | 0.00 |
| OE | A+B+C-paceshap | 0.36 | 0.39 | 0.46 | 0.49 | 0.21 | 0.00 |
| OE | A+B+C-oracle | 0.25 | 0.41 | 0.59 | 0.50 | 0.26 | 0.00 |
| random | A+B+C'-volume | 0.14 | 0.23 | 0.14 | 0.18 | 0.08 | 0.00 |
| margin | A+B+C'-volume | 0.21 | 0.22 | 0.07 | 0.23 | 0.09 | 0.00 |
| knn | A+B+C'-volume | 0.17 | 0.49 | 0.26 | 0.41 | 0.21 | 0.00 |
| oracle_errors | A+B+C'-volume | 0.28 | 0.41 | 0.39 | 0.36 | 0.22 | 0.00 |
| SR-300 | 0.32 | 0.23 | 0.23 | 0.29 | 0.13 | 0.00 |
| SR-300-train | 0.32 | 0.22 | 0.10 | 0.30 | 0.16 | 0.00 |
| SR-all | 0.32 | 0.23 | 0.23 | 0.63 | 0.29 | 0.00 |
| SR-all + OE | A+B+C'-volume | 0.25 | 0.41 | 0.48 | 0.61 | 0.31 | 0.00 |

### E4, box-level certificate

### Changes outside it (must be 0)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 0 | 0 | 0 | 0 | 0 | 0 |
| OE | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |
| OE | A+B+C | 0 | 0 | 0 | 0 | 0 | 0 |
| OE | A+B+C-fewest | 0 | 0 | 0 | 0 | 0 | 0 |
| OE | A+B+C-paceshap | 0 | 0 | 0 | 0 | 0 | 0 |
| OE | A+B+C-oracle | 0 | 0 | 0 | 0 | 0 | 0 |
| random | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |
| margin | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |
| knn | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |
| oracle_errors | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |
| SR-300 | 0 | 0 | 0 | 0 | 0 | 0 |
| SR-300-train | 0 | 0 | 0 | 0 | 0 | 0 |
| SR-all | 0 | 0 | 0 | 0 | 0 | 0 |
| SR-all + OE | A+B+C'-volume | 0 | 0 | 0 | 0 | 0 | 0 |

### Size per edit (% of audit rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 29.54 | 4.77 | 62.96 | 15.69 | 1.66 | 9.83 |
| OE | A+B+C'-volume | 26.52 | 5.30 | 60.75 | 16.85 | 2.20 | 7.95 |
| OE | A+B+C | 29.70 | 5.28 | 60.76 | 14.66 | 2.23 | 8.09 |
| OE | A+B+C-fewest | 31.93 | 5.27 | 61.75 | 16.89 | 2.36 | 7.56 |
| OE | A+B+C-paceshap | 24.55 | 4.81 | 62.11 | 14.33 | 2.04 | 7.43 |
| OE | A+B+C-oracle | 25.68 | 4.94 | 61.33 | 12.99 | 2.43 | 7.21 |
| random | A+B+C'-volume | 37.77 | 5.40 | 72.07 | 15.90 | 1.77 | 21.06 |
| margin | A+B+C'-volume | 39.45 | 7.31 | 71.93 | 19.13 | 2.13 | 9.66 |
| knn | A+B+C'-volume | 32.12 | 4.69 | 53.74 | 17.20 | 2.37 | 7.87 |
| oracle_errors | A+B+C'-volume | 37.24 | 5.23 | 65.69 | 16.12 | 2.17 | 7.64 |
| SR-300 | 23.22 | 4.07 | 16.32 | 8.57 | 1.55 | 3.95 |
| SR-300-train | 23.31 | 4.13 | 19.48 | 8.61 | 1.62 | 3.94 |
| SR-all | 23.22 | 4.04 | 16.32 | 5.42 | 1.28 | 3.66 |
| SR-all + OE | A+B+C'-volume | 23.34 | 4.02 | 56.70 | 5.61 | 1.31 | 5.56 |

### Precision: changed rows / certified rows, over all edits (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 0.0 | 0.5 | 0.0 | 0.1 | 0.9 | 0.0 |
| OE | A+B+C'-volume | 0.0 | 0.9 | 0.1 | 0.7 | 2.3 | 0.4 |
| OE | A+B+C | 0.0 | 1.3 | 0.1 | 1.5 | 2.6 | 0.4 |
| OE | A+B+C-fewest | 0.0 | 0.8 | 0.1 | 0.5 | 1.9 | 0.4 |
| OE | A+B+C-paceshap | 0.0 | 1.3 | 0.1 | 1.4 | 2.5 | 0.4 |
| OE | A+B+C-oracle | 0.0 | 1.8 | 0.1 | 1.2 | 3.9 | 0.3 |
| random | A+B+C'-volume | 0.0 | 0.4 | 0.0 | 0.4 | 1.1 | 0.1 |
| margin | A+B+C'-volume | 0.0 | 0.6 | 0.0 | 0.4 | 1.4 | 0.2 |
| knn | A+B+C'-volume | 0.0 | 1.0 | 0.1 | 0.7 | 2.3 | 0.5 |
| oracle_errors | A+B+C'-volume | 0.0 | 1.2 | 0.1 | 1.4 | 2.6 | 0.5 |
| SR-300 | 0.4 | 2.5 | 3.6 | 2.5 | 4.9 | 5.6 |
| SR-300-train | 0.4 | 2.5 | 2.3 | 2.5 | 4.8 | 5.6 |
| SR-all | 0.4 | 2.5 | 3.6 | 2.8 | 4.3 | 3.0 |
| SR-all + OE | A+B+C'-volume | 0.0 | 1.6 | 0.1 | 2.6 | 3.8 | 0.6 |

### Rows some edit could have changed (% of audit rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 98.8 | 64.1 | 98.8 | 88.6 | 66.4 | 49.6 |
| OE | A+B+C'-volume | 98.6 | 68.2 | 99.0 | 91.0 | 75.4 | 60.7 |
| OE | A+B+C | 99.1 | 67.6 | 98.8 | 91.0 | 76.9 | 57.0 |
| OE | A+B+C-fewest | 98.7 | 68.4 | 99.1 | 91.0 | 77.2 | 56.2 |
| OE | A+B+C-paceshap | 99.2 | 69.8 | 98.9 | 91.3 | 74.8 | 56.0 |
| OE | A+B+C-oracle | 99.4 | 70.1 | 98.7 | 91.0 | 74.9 | 48.3 |
| random | A+B+C'-volume | 98.4 | 87.8 | 99.1 | 90.9 | 82.9 | 97.7 |
| margin | A+B+C'-volume | 94.8 | 65.2 | 97.1 | 88.6 | 78.2 | 42.2 |
| knn | A+B+C'-volume | 96.7 | 72.4 | 99.5 | 90.7 | 77.8 | 46.2 |
| oracle_errors | A+B+C'-volume | 98.8 | 86.9 | 99.3 | 90.9 | 77.5 | 58.1 |
| SR-300 | 86.3 | 71.1 | 79.1 | 89.6 | 62.1 | 25.9 |
| SR-300-train | 86.1 | 70.5 | 69.4 | 89.6 | 66.2 | 25.9 |
| SR-all | 86.3 | 71.1 | 79.1 | 92.2 | 84.4 | 28.3 |
| SR-all + OE | A+B+C'-volume | 99.0 | 77.8 | 98.5 | 93.3 | 88.8 | 45.1 |

### Explained rows outside it (of 100)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 1 | 37 | 2 | 11 | 33 | 52 |
| OE | A+B+C'-volume | 1 | 34 | 2 | 8 | 24 | 38 |
| OE | A+B+C | 0 | 34 | 2 | 8 | 23 | 43 |
| OE | A+B+C-fewest | 1 | 32 | 2 | 9 | 23 | 43 |
| OE | A+B+C-paceshap | 1 | 32 | 2 | 8 | 23 | 44 |
| OE | A+B+C-oracle | 0 | 30 | 2 | 9 | 25 | 49 |
| random | A+B+C'-volume | 1 | 14 | 1 | 10 | 17 | 2 |
| margin | A+B+C'-volume | 6 | 36 | 3 | 12 | 20 | 56 |
| knn | A+B+C'-volume | 3 | 27 | 1 | 9 | 22 | 51 |
| oracle_errors | A+B+C'-volume | 1 | 13 | 1 | 10 | 26 | 41 |
| SR-300 | 13 | 29 | 25 | 11 | 36 | 72 |
| SR-300-train | 13 | 29 | 33 | 11 | 31 | 72 |
| SR-all | 13 | 29 | 25 | 7 | 14 | 68 |
| SR-all + OE | A+B+C'-volume | 1 | 22 | 2 | 5 | 10 | 55 |

### PACE-flip top-3 changed on those rows (share)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 0.00 | 0.15 | 0.00 | 0.07 | 0.08 | 0.00 |
| OE | A+B+C'-volume | 0.00 | 0.11 | 0.00 | 0.25 | 0.15 | 0.00 |
| OE | A+B+C | 0.00 | 0.16 | 0.00 | 0.23 | 0.15 | 0.00 |
| OE | A+B+C-fewest | 0.00 | 0.16 | 0.00 | 0.21 | 0.07 | 0.00 |
| OE | A+B+C-paceshap | 0.00 | 0.22 | 0.10 | 0.33 | 0.13 | 0.00 |
| OE | A+B+C-oracle | 0.00 | 0.14 | 0.00 | 0.26 | 0.13 | 0.00 |
| random | A+B+C'-volume | 0.00 | 0.11 | 0.00 | 0.04 | 0.04 | 0.00 |
| margin | A+B+C'-volume | 0.00 | 0.10 | 0.00 | 0.04 | 0.07 | 0.00 |
| knn | A+B+C'-volume | 0.00 | 0.36 | 0.00 | 0.19 | 0.12 | 0.00 |
| oracle_errors | A+B+C'-volume | 0.00 | 0.31 | 0.00 | 0.10 | 0.14 | 0.00 |
| SR-300 | 0.04 | 0.09 | 0.04 | 0.04 | 0.07 | 0.00 |
| SR-300-train | 0.05 | 0.09 | 0.00 | 0.04 | 0.08 | 0.00 |
| SR-all | 0.04 | 0.09 | 0.04 | 0.28 | 0.17 | 0.00 |
| SR-all + OE | A+B+C'-volume | 0.00 | 0.16 | 0.00 | 0.17 | 0.28 | 0.00 |

### Audit rows whose prediction one edit changed (mean)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE | A+B | 0.03 | 0.14 | 0.08 | 0.30 | 0.28 | 0.02 |
| OE | A+B+C'-volume | 0.06 | 0.32 | 0.15 | 2.35 | 1.02 | 0.60 |
| OE | A+B+C | 0.06 | 0.46 | 0.17 | 4.26 | 1.15 | 0.66 |
| OE | A+B+C-fewest | 0.04 | 0.27 | 0.23 | 1.68 | 0.87 | 0.53 |
| OE | A+B+C-paceshap | 0.07 | 0.41 | 0.22 | 3.83 | 1.02 | 0.65 |
| OE | A+B+C-oracle | 0.07 | 0.57 | 0.32 | 3.01 | 1.89 | 0.49 |
| random | A+B+C'-volume | 0.03 | 0.15 | 0.09 | 1.28 | 0.40 | 0.28 |
| margin | A+B+C'-volume | 0.04 | 0.27 | 0.13 | 1.49 | 0.58 | 0.43 |
| knn | A+B+C'-volume | 0.04 | 0.29 | 0.21 | 2.43 | 1.08 | 0.81 |
| oracle_errors | A+B+C'-volume | 0.05 | 0.40 | 0.20 | 4.20 | 1.12 | 0.81 |
| SR-300 | 0.72 | 0.65 | 2.63 | 4.08 | 1.51 | 4.59 |
| SR-300-train | 0.68 | 0.65 | 1.90 | 4.07 | 1.56 | 4.60 |
| SR-all | 0.72 | 0.65 | 2.63 | 2.86 | 1.11 | 2.24 |
| SR-all + OE | A+B+C'-volume | 0.07 | 0.42 | 0.30 | 2.73 | 0.99 | 0.64 |

### PACE-flip top-3 changed, all 100 explained rows (share)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B9 refit, all pool labels | 0.15 | 0.61 | 0.57 | 0.46 | 0.48 | 0.00 |
| OE | A+B | 0.10 | 0.40 | 0.41 | 0.18 | 0.18 | 0.00 |
| OE | A+B+C'-volume | 0.23 | 0.47 | 0.45 | 0.67 | 0.39 | 0.00 |
| OE | A+B+C | 0.30 | 0.55 | 0.61 | 0.79 | 0.46 | 0.00 |
| OE | A+B+C-fewest | 0.20 | 0.48 | 0.69 | 0.60 | 0.37 | 0.00 |
| OE | A+B+C-paceshap | 0.39 | 0.51 | 0.57 | 0.78 | 0.43 | 0.00 |
| OE | A+B+C-oracle | 0.28 | 0.53 | 0.70 | 0.75 | 0.50 | 0.00 |
| random | A+B+C'-volume | 0.17 | 0.35 | 0.24 | 0.50 | 0.22 | 0.00 |
| margin | A+B+C'-volume | 0.23 | 0.36 | 0.26 | 0.53 | 0.26 | 0.00 |
| knn | A+B+C'-volume | 0.19 | 0.57 | 0.43 | 0.72 | 0.38 | 0.00 |
| oracle_errors | A+B+C'-volume | 0.31 | 0.54 | 0.52 | 0.69 | 0.36 | 0.00 |
| CFMNN refit, +bought | 0.10 | 0.43 | 0.46 | 0.18 | 0.21 | 0.00 |
| CFMNN refit, +bought +acceptance | 0.21 | 0.62 | 0.59 | 0.45 | 0.43 | 0.00 |
| SR-300 | 0.34 | 0.39 | 0.39 | 0.70 | 0.34 | 0.00 |
| SR-300-train | 0.35 | 0.38 | 0.25 | 0.69 | 0.38 | 0.00 |
| SR-all | 0.34 | 0.40 | 0.39 | 0.87 | 0.57 | 0.00 |
| SR-all + OE | A+B+C'-volume | 0.28 | 0.54 | 0.61 | 0.82 | 0.57 | 0.00 |

### kNN and RF: predictions changed against the same learner on train only (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| kNN, +bought | 0.60 | 2.12 | 0.97 | 0.89 | 1.38 | 0.00 |
| RF, +bought | 0.14 | 2.59 | 0.73 | 2.37 | 2.42 | 0.00 |
| kNN, +bought +acceptance | 0.93 | 3.11 | 1.22 | 2.93 | 3.88 | 0.03 |
| RF, +bought +acceptance | 0.19 | 2.85 | 0.67 | 2.65 | 2.87 | 0.03 |

