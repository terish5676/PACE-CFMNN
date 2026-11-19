# Refinement v2: confirmatory run

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Seeds 50–59 (10 seeds), 6 datasets. Paired Wilcoxon over seeds, Holm across the datasets of each endpoint; `*` = Holm-adjusted p < 0.05. Two-sided, except the non-inferiority endpoints (S3b, S3c, S3d), which are one-sided tests of first − second > −margin (macro-F1 margin 1 point, lowest class recall margin 2 points). Written by `scripts/run_refinement_v2.py`.

## Endpoints

Mean difference (first arm − second arm) over seeds:

| Endpoint | Question | Value | First − second | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|---|---|---|
| P3 | Do bought labels add anything after self-repair? | test_macro_f1 | SR+AL − SR | +0.57 | +1.08* | +2.28 | +2.03* | +0.73* | +0.55 |
| S3a | Does self-repair (no new labels) help? | test_macro_f1 | SR − B0 | +6.02* | +1.48* | +2.58* | +3.27* | +2.25* | +5.66* |
| S3b | Is self-repair no worse than bought labels alone, within the margin? | test_macro_f1 | SR − AL | -0.36 | -0.47 | -2.30 | -1.80 | +0.33* | -0.47 |
| S3c | Is SR+AL's lowest class recall no worse than B0's, within the margin? | test_min_class_recall | SR+AL − B0 | +20.48* | +9.14* | -1.26 | -1.22 | +3.76* | +17.87* |
| S3d | Is SR's lowest class recall no worse than B0's, within the margin? | test_min_class_recall | SR − B0 | +19.37* | +6.41* | +0.08 | -2.73 | +2.93* | +16.65* |
| P4 | Of the predictions an update changes, are more of them fixes with repair than with retraining? | churn_fix_share | SR+AL − Retrain, shuffled | +33.46* | +11.86* | +39.38* | +17.61* | +15.32* | +53.60* |
| S4a | Does repair break fewer correct predictions than retraining? | churn_broken_share | Retrain, shuffled − SR+AL | +0.83* | +3.73* | +5.99* | +8.75* | +7.43* | +3.60* |
| S4b | Does repair change fewer explanations (PACE-flip top 3) than retraining? | expl_topk_change_share | Retrain, shuffled − SR+AL | +0.35* | +0.48* | +0.32* | +0.17* | +0.45* | – |
| S4c | Is repair more accurate than retraining with the same labels? | test_macro_f1 | SR+AL − Retrain, shuffled | +4.81* | +1.68* | +8.24* | +6.78* | +2.63* | +8.13* |

Wins–losses over seeds (n = seeds tested), raw p and Holm-adjusted p:

| Endpoint | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| P3 | 5–4 (n=10), p=0.359, Holm 0.387 | 8–2 (n=10), p=0.00977, Holm 0.0488 | 7–3 (n=10), p=0.193, Holm 0.387 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 9–1 (n=10), p=0.00977, Holm 0.0488 | 7–3 (n=10), p=0.0645, Holm 0.193 |
| S3a | 9–1 (n=10), p=0.00391, Holm 0.0117 | 7–3 (n=10), p=0.0488, Holm 0.0488 | 8–2 (n=10), p=0.00977, Holm 0.0195 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S3b | 4–6 (n=10), p=0.216, Holm 0.647 | 4–6 (n=10), p=0.138, Holm 0.551 | 3–7 (n=10), p=0.784, Holm 1 | 0–10 (n=10), p=0.986, Holm 1 | 6–4 (n=10), p=0.000977, Holm 0.00586 | 4–6 (n=10), p=0.0244, Holm 0.122 |
| S3c | 8–1 (n=10), p=0.00195, Holm 0.00977 | 9–1 (n=10), p=0.00195, Holm 0.00977 | 1–2 (n=10), p=0.0605, Holm 0.121 | 5–5 (n=10), p=0.216, Holm 0.216 | 8–2 (n=10), p=0.00293, Holm 0.00977 | 10–0 (n=10), p=0.000977, Holm 0.00586 |
| S3d | 9–0 (n=10), p=0.000977, Holm 0.00586 | 7–2 (n=10), p=0.00684, Holm 0.0205 | 1–1 (n=10), p=0.0371, Holm 0.0742 | 1–9 (n=10), p=0.839, Holm 0.839 | 7–2 (n=10), p=0.00293, Holm 0.0117 | 10–0 (n=10), p=0.000977, Holm 0.00586 |
| P4 | 9–1 (n=10), p=0.00391, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S4a | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S4b | 10–0 (n=10), p=0.00195, Holm 0.00977 | 10–0 (n=10), p=0.00195, Holm 0.00977 | 10–0 (n=10), p=0.00195, Holm 0.00977 | 10–0 (n=10), p=0.00195, Holm 0.00977 | 10–0 (n=10), p=0.00195, Holm 0.00977 | not tested |
| S4c | 9–1 (n=10), p=0.00586, Holm 0.0117 | 9–1 (n=10), p=0.00586, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 9–1 (n=10), p=0.00391, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |

## Certificate check (C4)

Test-row changes outside the tight certificate, summed over 180 edit-based runs: **0** (must be 0); outside the box-level certificate: **0** (must be 0). Accepted edits checked: 46,153.

## Share of the gap to RF closed (macro-F1, %)

100 × (arm − B0) / (RF with every label − B0), from the means over seeds.

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| SR | 44 | 20 | 17 | 23 | 10 | 83 |
| AL | 47 | 26 | 33 | 36 | 9 | 90 |
| SR+AL | 48 | 34 | 33 | 37 | 14 | 91 |
| Retrain, shuffled | 13 | 12 | -23 | -10 | 2 | -28 |
| Retrain, appended | 3 | 4 | -18 | -2 | 4 | -7 |
| B9 | 1 | -0 | -17 | -5 | 4 | -5 |

## Test macro-F1 (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 84.0 | 82.0 | 67.6 | 72.6 | 74.3 | 93.1 |
| B9 | 84.1 | 82.0 | 65.1 | 71.9 | 75.1 | 92.7 |
| AL | 90.3 | 83.9 | 72.4 | 77.7 | 76.3 | 99.2 |
| SR | 90.0 | 83.5 | 70.1 | 75.9 | 76.6 | 98.8 |
| SR+AL | 90.5 | 84.5 | 72.4 | 77.9 | 77.3 | 99.3 |
| Retrain, appended | 84.4 | 82.3 | 65.0 | 72.3 | 75.2 | 92.6 |
| Retrain, shuffled | 85.7 | 82.9 | 64.2 | 71.1 | 74.7 | 91.2 |
| kNN, training rows | 54.9 | 87.2 | 63.8 | 80.0 | 92.3 | 99.9 |
| kNN, every label | 58.4 | 87.9 | 68.2 | 80.2 | 93.5 | 99.9 |
| RF, training rows | 96.9 | 88.7 | 80.4 | 86.4 | 95.2 | 99.9 |
| RF, every label | 97.7 | 89.6 | 82.4 | 86.8 | 95.9 | 99.9 |

## Lowest class recall (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 59.2 | 47.3 | 52.4 | 71.9 | 49.7 | 80.9 |
| B9 | 59.7 | 46.2 | 48.0 | 71.1 | 50.9 | 80.0 |
| AL | 78.4 | 55.3 | 50.6 | 72.2 | 51.6 | 98.3 |
| SR | 78.6 | 53.7 | 52.5 | 69.1 | 52.6 | 97.6 |
| SR+AL | 79.7 | 56.4 | 51.2 | 70.6 | 53.4 | 98.8 |
| Retrain, appended | 57.3 | 46.2 | 53.0 | 72.2 | 51.3 | 79.7 |
| Retrain, shuffled | 62.2 | 49.8 | 53.7 | 70.0 | 49.3 | 76.5 |
| kNN, training rows | 5.2 | 64.6 | 10.8 | 61.0 | 80.7 | 99.9 |
| kNN, every label | 8.4 | 66.2 | 19.2 | 61.4 | 83.0 | 99.9 |
| RF, training rows | 91.8 | 60.8 | 48.3 | 76.9 | 86.4 | 99.9 |
| RF, every label | 94.5 | 62.4 | 51.9 | 77.7 | 88.5 | 99.9 |

## Test accuracy (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 97.0 | 84.8 | 88.8 | 74.0 | 74.5 | 95.8 |
| B9 | 97.0 | 84.8 | 87.6 | 73.2 | 75.3 | 95.6 |
| AL | 98.1 | 86.3 | 93.0 | 79.5 | 76.5 | 99.5 |
| SR | 98.0 | 86.0 | 90.8 | 77.9 | 76.8 | 99.2 |
| SR+AL | 98.2 | 86.9 | 93.4 | 79.9 | 77.5 | 99.5 |
| Retrain, appended | 97.0 | 85.2 | 87.7 | 73.7 | 75.4 | 95.6 |
| Retrain, shuffled | 97.2 | 85.5 | 87.2 | 72.3 | 75.0 | 94.8 |
| kNN, training rows | 93.7 | 89.2 | 95.0 | 83.3 | 92.3 | 99.9 |
| kNN, every label | 93.9 | 89.8 | 95.2 | 83.4 | 93.5 | 99.9 |
| RF, training rows | 99.5 | 90.8 | 96.8 | 87.9 | 95.2 | 99.9 |
| RF, every label | 99.6 | 91.5 | 97.1 | 88.3 | 95.9 | 100.0 |

## Labels used beyond the training rows

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B9 | 1,440 | 1,287 | 1,094 | 3,804 | 4,000 | 49,011 |
| AL | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| SR | 0 | 0 | 0 | 0 | 0 | 0 |
| SR+AL | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| Retrain, appended | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| Retrain, shuffled | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| kNN, training rows | 0 | 0 | 0 | 0 | 0 | 0 |
| kNN, every label | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| RF, training rows | 0 | 0 | 0 | 0 | 0 | 0 |
| RF, every label | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |

## Test predictions changed against B0 (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B9 | 0.58 | 3.90 | 3.98 | 7.39 | 9.75 | 0.61 |
| AL | 1.42 | 5.34 | 6.35 | 15.78 | 8.75 | 4.13 |
| SR | 1.46 | 6.51 | 3.92 | 18.40 | 10.64 | 4.46 |
| SR+AL | 1.62 | 7.41 | 6.79 | 19.57 | 13.46 | 4.18 |
| Retrain, appended | 0.83 | 4.36 | 5.55 | 5.58 | 8.00 | 0.73 |
| Retrain, shuffled | 2.36 | 14.01 | 12.63 | 29.47 | 30.00 | 6.61 |
| kNN, training rows | 4.56 | 13.99 | 11.61 | 25.21 | 25.66 | 4.17 |
| kNN, every label | 4.47 | 14.18 | 11.64 | 25.01 | 25.64 | 4.17 |
| RF, training rows | 2.96 | 12.67 | 10.77 | 23.52 | 25.08 | 4.17 |
| RF, every label | 2.99 | 12.47 | 10.82 | 23.42 | 25.12 | 4.18 |

## Test predictions broken, right → wrong, against B0 (% of test rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B9 | 0.28 | 1.77 | 2.50 | 4.12 | 2.93 | 0.41 |
| AL | 0.12 | 1.69 | 0.82 | 5.16 | 1.99 | 0.22 |
| SR | 0.19 | 2.42 | 0.73 | 7.25 | 2.35 | 0.54 |
| SR+AL | 0.19 | 2.36 | 0.82 | 6.85 | 3.06 | 0.22 |
| Retrain, appended | 0.39 | 1.77 | 3.23 | 2.97 | 2.33 | 0.49 |
| Retrain, shuffled | 1.03 | 6.09 | 6.81 | 15.60 | 10.48 | 3.83 |
| kNN, training rows | 3.86 | 4.32 | 2.50 | 7.97 | 2.34 | 0.03 |
| kNN, every label | 3.74 | 4.10 | 2.39 | 7.80 | 1.97 | 0.03 |
| RF, training rows | 0.21 | 2.93 | 1.22 | 4.82 | 1.16 | 0.02 |
| RF, every label | 0.18 | 2.52 | 1.09 | 4.60 | 0.96 | 0.02 |

## Test predictions fixed, wrong → right, against B0 (% of test rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B9 | 0.28 | 1.83 | 1.30 | 3.28 | 3.75 | 0.20 |
| AL | 1.26 | 3.23 | 5.05 | 10.63 | 3.92 | 3.90 |
| SR | 1.26 | 3.62 | 2.76 | 11.15 | 4.59 | 3.92 |
| SR+AL | 1.40 | 4.49 | 5.40 | 12.73 | 6.00 | 3.95 |
| Retrain, appended | 0.42 | 2.17 | 2.15 | 2.61 | 3.20 | 0.23 |
| Retrain, shuffled | 1.22 | 6.77 | 5.20 | 13.87 | 10.90 | 2.79 |
| kNN, training rows | 0.58 | 8.71 | 8.69 | 17.24 | 20.07 | 4.14 |
| kNN, every label | 0.62 | 9.10 | 8.80 | 17.21 | 20.92 | 4.15 |
| RF, training rows | 2.72 | 8.96 | 9.25 | 18.70 | 21.79 | 4.15 |
| RF, every label | 2.79 | 9.22 | 9.42 | 18.83 | 22.29 | 4.16 |

## Fix share: fixed / (fixed + broken), against B0 (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B9 | 54.2 | 51.2 | 31.8 | 44.0 | 56.2 | 41.9 |
| AL | 85.5 | 64.6 | 75.6 | 67.0 | 66.1 | 94.6 |
| SR | 86.4 | 59.0 | 69.4 | 60.3 | 65.9 | 87.5 |
| SR+AL | 85.3 | 64.2 | 81.9 | 64.8 | 66.2 | 94.1 |
| Retrain, appended | 51.2 | 55.0 | 32.0 | 46.4 | 57.8 | 44.4 |
| Retrain, shuffled | 51.8 | 52.4 | 42.6 | 47.2 | 50.9 | 40.5 |
| kNN, training rows | 14.1 | 66.8 | 75.7 | 68.2 | 89.6 | 99.2 |
| kNN, every label | 15.9 | 68.9 | 76.6 | 68.7 | 91.4 | 99.2 |
| RF, training rows | 93.6 | 75.6 | 87.2 | 79.5 | 95.0 | 99.4 |
| RF, every label | 93.6 | 78.9 | 88.4 | 80.3 | 95.9 | 99.5 |

## PACE-flip top-3 changed against B0, 100 test rows (share)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B9 | 0.18 | 0.58 | 0.45 | 0.50 | 0.50 | 0.00 |
| AL | 0.24 | 0.40 | 0.50 | 0.69 | 0.37 | 0.00 |
| SR | 0.25 | 0.33 | 0.35 | 0.72 | 0.39 | 0.00 |
| SR+AL | 0.26 | 0.50 | 0.61 | 0.78 | 0.47 | 0.00 |
| Retrain, appended | 0.23 | 0.54 | 0.55 | 0.42 | 0.42 | 0.00 |
| Retrain, shuffled | 0.61 | 0.98 | 0.93 | 0.95 | 0.92 | 0.00 |

## PACE-flip top feature changed against B0, 100 test rows (share)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B9 | 0.07 | 0.37 | 0.26 | 0.27 | 0.24 | 0.03 |
| AL | 0.12 | 0.28 | 0.30 | 0.43 | 0.21 | 0.11 |
| SR | 0.09 | 0.25 | 0.22 | 0.50 | 0.24 | 0.11 |
| SR+AL | 0.13 | 0.33 | 0.40 | 0.52 | 0.27 | 0.11 |
| Retrain, appended | 0.10 | 0.35 | 0.29 | 0.23 | 0.22 | 0.03 |
| Retrain, shuffled | 0.29 | 0.87 | 0.64 | 0.66 | 0.63 | 0.18 |

## Training error at the end (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 1.7 | 9.4 | 10.4 | 24.5 | 20.2 | 4.2 |
| B9 | 1.8 | 9.9 | 11.7 | 25.5 | 20.7 | 4.4 |
| AL | 0.8 | 8.2 | 6.8 | 19.1 | 18.3 | 0.5 |
| SR | 0.4 | 6.0 | 8.4 | 19.7 | 16.6 | 0.8 |
| SR+AL | 0.6 | 7.1 | 6.5 | 18.0 | 16.2 | 0.5 |
| Retrain, appended | 2.0 | 9.8 | 11.4 | 24.9 | 20.5 | 4.4 |
| Retrain, shuffled | 1.9 | 9.8 | 12.7 | 25.7 | 20.8 | 5.3 |

## Run time (s)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B9 | 1 | 1 | 0 | 1 | 2 | 14 |
| AL | 35 | 122 | 17 | 168 | 538 | 657 |
| SR | 3 | 148 | 7 | 204 | 1030 | 92 |
| SR+AL | 35 | 118 | 16 | 171 | 541 | 612 |
| Retrain, appended | 0 | 1 | 0 | 1 | 2 | 11 |
| Retrain, shuffled | 0 | 1 | 0 | 1 | 2 | 11 |
| kNN, training rows | 0 | 0 | 0 | 0 | 0 | 1 |
| kNN, every label | 0 | 0 | 0 | 0 | 0 | 1 |
| RF, training rows | 1 | 1 | 1 | 5 | 3 | 9 |
| RF, every label | 1 | 2 | 1 | 6 | 3 | 11 |

## Certificates (edit-based arms; 2,000 test rows)

SR+AL's certificates cover all its edits, self-repair included.

### tight: changes outside it (must be 0)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| AL | 0 | 0 | 0 | 0 | 0 | 0 |
| SR | 0 | 0 | 0 | 0 | 0 | 0 |
| SR+AL | 0 | 0 | 0 | 0 | 0 | 0 |

### tight: size per edit (% of rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| AL | 1.24 | 2.74 | 40.52 | 13.34 | 1.92 | 5.41 |
| SR | 1.23 | 2.03 | 20.04 | 7.64 | 1.42 | 3.08 |
| SR+AL | 1.19 | 1.85 | 34.55 | 7.85 | 1.55 | 4.05 |

### tight: precision, changed / certified (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| AL | 0.8 | 1.8 | 0.1 | 1.0 | 2.7 | 0.5 |
| SR | 10.4 | 4.8 | 2.1 | 2.8 | 5.3 | 5.4 |
| SR+AL | 0.7 | 3.4 | 0.2 | 1.9 | 3.7 | 0.7 |

### tight: rows some edit could change (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| AL | 5.3 | 21.4 | 68.1 | 73.4 | 50.0 | 21.0 |
| SR | 4.7 | 19.9 | 64.2 | 72.6 | 47.7 | 20.3 |
| SR+AL | 5.4 | 21.7 | 68.6 | 74.0 | 51.5 | 20.9 |

### tight: PACE-flip top-3 changed outside it (share)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| AL | 0.22 | 0.29 | 0.29 | 0.38 | 0.20 | 0.00 |
| SR | 0.22 | 0.20 | 0.16 | 0.38 | 0.17 | 0.00 |
| SR+AL | 0.24 | 0.39 | 0.41 | 0.51 | 0.24 | 0.00 |

### loose: changes outside it (must be 0)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| AL | 0 | 0 | 0 | 0 | 0 | 0 |
| SR | 0 | 0 | 0 | 0 | 0 | 0 |
| SR+AL | 0 | 0 | 0 | 0 | 0 | 0 |

### loose: size per edit (% of rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| AL | 27.37 | 4.98 | 65.15 | 15.02 | 2.15 | 7.59 |
| SR | 27.31 | 3.99 | 28.61 | 8.43 | 1.59 | 3.81 |
| SR+AL | 27.03 | 3.94 | 61.59 | 9.55 | 1.75 | 6.56 |

### loose: precision, changed / certified (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| AL | 0.0 | 1.0 | 0.1 | 0.9 | 2.4 | 0.3 |
| SR | 0.4 | 2.4 | 1.4 | 2.5 | 4.8 | 4.3 |
| SR+AL | 0.0 | 1.6 | 0.1 | 1.6 | 3.3 | 0.4 |

### loose: rows some edit could change (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| AL | 98.9 | 67.6 | 99.0 | 90.4 | 74.8 | 54.0 |
| SR | 87.4 | 69.0 | 90.6 | 88.2 | 67.2 | 26.1 |
| SR+AL | 99.0 | 76.4 | 99.4 | 91.8 | 82.0 | 57.5 |

### loose: PACE-flip top-3 changed outside it (share)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| AL | 0.29 | 0.16 | 0.19 | 0.21 | 0.13 | 0.00 |
| SR | 0.00 | 0.08 | 0.10 | 0.14 | 0.12 | 0.00 |
| SR+AL | 0.40 | 0.15 | 0.21 | 0.24 | 0.13 | 0.00 |

## kNN and RF: predictions changed against the same learner on the training rows (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| kNN, every label | 0.75 | 3.43 | 0.89 | 3.06 | 3.58 | 0.01 |
| RF, every label | 0.21 | 2.92 | 0.86 | 2.68 | 3.01 | 0.02 |

