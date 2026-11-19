# Run F: self-repair against contraction

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). RQ3's split, seeds 120–129 (10), 6 datasets. Each comparison: paired two-sided Wilcoxon over seeds, Holm across the datasets; first arm − second arm. `*` = Holm-adjusted p < 0.05. Written by `scripts/run_contraction.py`.

Arms: B0 = the starting CFMNN (θ = 0.3, γ = 4); SR = self-repair; SC-all = Simpson's contraction on every overlap; SC-err = Simpson's contraction at the same up-to-300 training errors as SR, no gate; GC-all, GC-err = the same with GFMM's overlap test and contraction rule; GFMM = online GFMM, which contracts while it trains.

## Comparisons

Mean difference over seeds (first − second):

| Code | Question | Value | First − second | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin | Higher / no significant difference / lower |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 (primary) | Self-repair or Simpson's contraction on every overlap: which is more accurate? | test_macro_f1 | SR − SC-all | +4.47* | +2.44* | +9.97* | +14.93* | +2.64* | +22.46* | 6 / 0 / 0 |
| S1 | Self-repair or ungated contraction at the same training errors? | test_macro_f1 | SR − SC-err | +4.47* | +1.71* | +5.27* | +5.06* | +2.44* | +9.05* | 6 / 0 / 0 |
| S2a | Self-repair or GFMM's contraction rule on every overlap? | test_macro_f1 | SR − GC-all | +29.10* | +5.67* | +15.33* | +14.98* | +30.75* | +26.59* | 6 / 0 / 0 |
| S2b | Self-repair or GFMM's contraction rule at the same training errors? | test_macro_f1 | SR − GC-err | +22.18* | +3.49* | +13.76* | +4.77* | +4.29* | +12.78* | 6 / 0 / 0 |
| S3 | Self-repair or GFMM, which contracts while it trains? | test_macro_f1 | SR − GFMM | +14.43* | +1.47 | +1.99 | +1.99* | -5.39* | +1.00* | 3 / 2 / 1 |
| S4a | Does self-repair help on these seeds? | test_macro_f1 | SR − B0 | +4.47* | +1.95* | +6.28* | +1.92* | +2.44* | +7.60* | 6 / 0 / 0 |
| S4b | Does contraction on every overlap help? | test_macro_f1 | SC-all − B0 | +0.00 | -0.48 | -3.69 | -13.01* | -0.20 | -14.85* | 0 / 4 / 2 |
| S4c | Does contraction at 300 training errors help? | test_macro_f1 | SC-err − B0 | +0.00 | +0.24 | +1.01 | -3.14* | +0.00 | -1.44 | 0 / 5 / 1 |
| S4d | Does GFMM's rule on every overlap help? | test_macro_f1 | GC-all − B0 | -24.63* | -3.72* | -9.05 | -13.06* | -28.31* | -18.99* | 0 / 1 / 5 |
| S4f | Does GFMM's rule at 300 training errors help? | test_macro_f1 | GC-err − B0 | -17.71* | -1.54 | -7.48 | -2.85* | -1.85* | -5.17 | 0 / 3 / 3 |
| S4e | Is GFMM more accurate than the starting model? | test_macro_f1 | GFMM − B0 | -9.96* | +0.48 | +4.28 | -0.07 | +7.83* | +6.60* | 2 / 3 / 1 |
| S5 | Lowest class recall: self-repair or contraction on every overlap? | test_min_class_recall | SR − SC-all | +12.00* | +10.89* | +20.12* | +16.54* | +2.87* | +53.63* | 6 / 0 / 0 |
| S6 | Boxes at the end: self-repair or contraction on every overlap? | boxes_end | SR − SC-all | -0.80 | +0.40 | +0.60 | +40.30* | -0.30 | +1.20 | 1 / 5 / 0 |
| S7 | Test predictions broken (right to wrong): self-repair or contraction on every overlap? | test_broken_pct | SR − SC-all | +0.24 | +0.05 | -10.00* | -18.28* | +1.97* | -7.66* | 1 / 2 / 3 |
| S8 | Training rows evicted from their boxes: self-repair or contraction on every overlap? | rows_evicted_pct | SR − SC-all | +59.46* | +11.07* | -21.49 | -56.63* | +8.65* | -15.02* | 3 / 1 / 2 |
| S9 | P1 by accuracy | test_accuracy | SR − SC-all | +0.88* | +1.86* | +10.20 | +16.15* | +2.61* | +11.03* | 5 / 1 / 0 |

Wins–losses over seeds (n = seeds tested), raw p and Holm-adjusted p:

| Code | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| P1 (primary) | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 9–1 (n=10), p=0.0195, Holm 0.0195 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S1 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 8–2 (n=10), p=0.0195, Holm 0.0195 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S2a | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 9–1 (n=10), p=0.00586, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S2b | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 9–1 (n=10), p=0.00586, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S3 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 8–2 (n=10), p=0.0371, Holm 0.0742 | 4–6 (n=10), p=0.77, Holm 0.77 | 8–2 (n=10), p=0.00977, Holm 0.0293 | 0–10 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S4a | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S4b | 0–0 (n=10), p=1, Holm 1 | 4–6 (n=10), p=0.375, Holm 0.75 | 4–6 (n=10), p=0.232, Holm 0.697 | 0–10 (n=10), p=0.00195, Holm 0.0117 | 3–7 (n=10), p=0.084, Holm 0.336 | 1–9 (n=10), p=0.00391, Holm 0.0195 |
| S4c | 0–0 (n=10), p=1, Holm 1 | 7–2 (n=10), p=0.0977, Holm 0.488 | 6–3 (n=10), p=0.359, Holm 1 | 0–10 (n=10), p=0.00195, Holm 0.0117 | 0–0 (n=10), p=1, Holm 1 | 6–4 (n=10), p=1, Holm 1 |
| S4d | 0–10 (n=10), p=0.00195, Holm 0.0117 | 2–8 (n=10), p=0.00977, Holm 0.0195 | 2–8 (n=10), p=0.105, Holm 0.105 | 0–10 (n=10), p=0.00195, Holm 0.0117 | 0–10 (n=10), p=0.00195, Holm 0.0117 | 1–9 (n=10), p=0.00391, Holm 0.0117 |
| S4f | 0–10 (n=10), p=0.00195, Holm 0.0117 | 2–8 (n=10), p=0.0195, Holm 0.0586 | 2–8 (n=10), p=0.16, Holm 0.16 | 0–10 (n=10), p=0.00195, Holm 0.0117 | 1–9 (n=10), p=0.00391, Holm 0.0156 | 2–8 (n=10), p=0.0645, Holm 0.129 |
| S4e | 0–10 (n=10), p=0.00195, Holm 0.0117 | 7–3 (n=10), p=0.375, Holm 0.75 | 8–2 (n=10), p=0.131, Holm 0.393 | 4–6 (n=10), p=0.77, Holm 0.77 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S5 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 7–0 (n=10), p=0.0156, Holm 0.0469 | 7–0 (n=10), p=0.0156, Holm 0.0469 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 6–2 (n=10), p=0.0391, Holm 0.0469 | 10–0 (n=10), p=0.00195, Holm 0.0117 |
| S6 | 0–4 (n=10), p=0.125, Holm 0.5 | 4–0 (n=10), p=0.125, Holm 0.5 | 4–0 (n=10), p=0.125, Holm 0.5 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 0–3 (n=10), p=0.25, Holm 0.5 | 7–0 (n=10), p=0.0156, Holm 0.0781 |
| S7 | 6–0 (n=10), p=0.0312, Holm 0.0625 | 6–4 (n=10), p=0.922, Holm 0.922 | 1–9 (n=10), p=0.00586, Holm 0.0176 | 0–10 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 0–10 (n=10), p=0.00195, Holm 0.0117 |
| S8 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 2–8 (n=10), p=0.0645, Holm 0.0645 | 0–10 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 0–10 (n=10), p=0.00195, Holm 0.0117 |
| S9 | 9–0 (n=10), p=0.00391, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 8–2 (n=10), p=0.0879, Holm 0.0879 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 | 10–0 (n=10), p=0.00195, Holm 0.0117 |

Means over seeds follow. A blank cell means the value does not apply to that arm.

## Test macro-F1 (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 85.7 | 82.6 | 61.8 | 72.5 | 74.5 | 91.3 |
| SR | 90.1 | 84.5 | 68.1 | 74.4 | 76.9 | 98.9 |
| SC-all | 85.7 | 82.1 | 58.1 | 59.5 | 74.3 | 76.5 |
| SC-err | 85.7 | 82.8 | 62.8 | 69.3 | 74.5 | 89.9 |
| GC-all | 61.0 | 78.9 | 52.8 | 59.4 | 46.2 | 72.4 |
| GC-err | 67.9 | 81.0 | 54.3 | 69.6 | 72.6 | 86.2 |
| GFMM | 75.7 | 83.1 | 66.1 | 72.4 | 82.3 | 97.9 |

## Test accuracy (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 97.2 | 85.1 | 86.9 | 73.9 | 74.7 | 94.8 |
| SR | 98.1 | 86.6 | 90.7 | 76.5 | 77.1 | 99.3 |
| SC-all | 97.2 | 84.8 | 80.5 | 60.3 | 74.5 | 88.3 |
| SC-err | 97.2 | 85.4 | 86.8 | 70.5 | 74.7 | 94.2 |
| GC-all | 94.2 | 84.1 | 70.7 | 60.3 | 52.0 | 86.7 |
| GC-err | 94.7 | 84.6 | 71.6 | 71.0 | 73.0 | 92.3 |
| GFMM | 95.6 | 85.8 | 93.4 | 74.7 | 82.4 | 98.7 |

## Lowest class recall (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 65.1 | 50.8 | 43.1 | 71.9 | 49.8 | 76.6 |
| SR | 77.1 | 59.9 | 49.0 | 68.4 | 52.6 | 97.8 |
| SC-all | 65.1 | 49.0 | 28.9 | 51.8 | 49.7 | 44.2 |
| SC-err | 65.1 | 51.0 | 40.4 | 66.6 | 49.8 | 73.2 |
| GC-all | 8.2 | 25.8 | 24.8 | 51.6 | 0.0 | 36.9 |
| GC-err | 19.3 | 37.6 | 28.8 | 67.7 | 40.0 | 64.3 |
| GFMM | 49.5 | 49.1 | 30.0 | 65.4 | 59.1 | 94.8 |

## Boxes at the end

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 46.8 | 151.0 | 23.6 | 227.8 | 450.4 | 80.8 |
| SR | 46.0 | 151.0 | 23.6 | 227.8 | 450.1 | 80.8 |
| SC-all | 46.8 | 150.6 | 23.0 | 187.5 | 450.4 | 79.6 |
| SC-err | 46.8 | 151.0 | 23.6 | 227.0 | 450.4 | 80.8 |
| GC-all | 42.5 | 122.2 | 21.4 | 188.3 | 116.1 | 79.5 |
| GC-err | 46.0 | 149.3 | 22.9 | 226.5 | 448.3 | 80.6 |
| GFMM | 299.1 | 223.6 | 76.5 | 275.3 | 638.6 | 65.5 |

## Training rows evicted from their boxes (% of training rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SR | 59.46 | 23.54 | 5.80 | 32.65 | 9.50 | 2.78 |
| SC-all | 0.00 | 12.47 | 27.28 | 89.28 | 0.85 | 17.80 |
| SC-err | 0.00 | 2.73 | 3.06 | 62.53 | 0.00 | 8.65 |
| GC-all | 59.64 | 62.24 | 69.66 | 89.56 | 89.44 | 21.09 |
| GC-err | 25.53 | 29.20 | 62.46 | 62.98 | 27.28 | 12.33 |

## Training rows evicted from their boxes (count)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| SR | 2,140.4 | 757.2 | 158.6 | 3,104.7 | 949.6 | 3,406.1 |
| SC-all | 0.0 | 401.2 | 746.5 | 8,490.5 | 84.7 | 21,813.9 |
| SC-err | 0.0 | 87.8 | 83.8 | 5,946.4 | 0.0 | 10,599.9 |
| GC-all | 2,147.0 | 2,002.3 | 1,906.0 | 8,516.7 | 8,944.4 | 25,844.3 |
| GC-err | 919.2 | 939.3 | 1,709.0 | 5,989.4 | 2,727.9 | 15,107.7 |

## Training rows inside a box of another class, at the end (count)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 196.5 | 790.2 | 1,780.0 | 7,210.6 | 5,154.4 | 24,161.5 |
| SR | 125.3 | 540.8 | 1,396.4 | 6,137.8 | 4,809.4 | 19,257.3 |
| SC-all | 196.5 | 719.3 | 1,075.0 | 1,548.2 | 5,137.6 | 16,455.8 |
| SC-err | 196.5 | 770.3 | 1,432.8 | 4,057.2 | 5,154.4 | 19,292.5 |
| GC-all | 124.8 | 126.0 | 269.7 | 1,549.4 | 89.6 | 13,015.0 |
| GC-err | 162.8 | 333.1 | 445.3 | 3,754.0 | 4,504.0 | 16,875.0 |

## Training error at the end (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 1.66 | 9.69 | 12.25 | 23.52 | 20.62 | 5.12 |
| SR | 0.52 | 6.28 | 8.53 | 20.19 | 16.68 | 0.65 |
| SC-all | 1.66 | 10.03 | 18.42 | 39.00 | 20.66 | 11.73 |
| SC-err | 1.66 | 9.63 | 12.40 | 27.65 | 20.62 | 5.83 |
| GC-all | 5.28 | 13.48 | 28.45 | 39.36 | 47.54 | 13.27 |
| GC-err | 4.18 | 10.81 | 27.99 | 27.63 | 22.30 | 7.63 |
| GFMM | 0.89 | 7.19 | 4.30 | 23.06 | 9.64 | 1.31 |

## Test predictions fixed, wrong → right, against B0 (% of test rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SR | 1.11 | 3.65 | 5.16 | 10.57 | 4.84 | 4.80 |
| SC-all | 0.00 | 1.74 | 4.96 | 12.70 | 0.27 | 1.43 |
| SC-err | 0.00 | 0.48 | 2.45 | 10.99 | 0.00 | 2.43 |
| GC-all | 0.51 | 5.08 | 6.50 | 12.83 | 5.45 | 1.09 |
| GC-err | 0.56 | 3.82 | 6.17 | 11.36 | 5.34 | 2.26 |
| GFMM | 1.08 | 6.96 | 10.49 | 15.30 | 15.03 | 4.61 |

## Test predictions broken, right → wrong, against B0 (% of test rows)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SR | 0.24 | 2.13 | 1.39 | 8.04 | 2.44 | 0.34 |
| SC-all | 0.00 | 2.08 | 11.39 | 26.31 | 0.47 | 7.99 |
| SC-err | 0.00 | 0.25 | 2.50 | 14.45 | 0.00 | 3.12 |
| GC-all | 3.58 | 6.13 | 22.68 | 26.50 | 28.14 | 9.21 |
| GC-err | 3.11 | 4.33 | 21.41 | 14.33 | 7.02 | 4.77 |
| GFMM | 2.75 | 6.26 | 4.00 | 14.53 | 7.28 | 0.78 |

## Edits or contractions kept

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| SR | 14.0 | 175.4 | 12.3 | 191.7 | 196.6 | 28.7 |
| SC-all | 0.0 | 66.8 | 12.5 | 1,120.9 | 9.0 | 50.5 |
| SC-err | 0.0 | 5.2 | 3.0 | 97.7 | 0.0 | 12.1 |
| GC-all | 27.1 | 659.1 | 35.2 | 1,131.6 | 6,839.6 | 61.1 |
| GC-err | 9.8 | 69.5 | 12.4 | 102.8 | 196.1 | 16.2 |

## Boxes removed because no training row was left in them

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| SR | 0.8 | 0.0 | 0.0 | 0.0 | 0.3 | 0.0 |
| SC-all | 0.0 | 0.4 | 0.6 | 40.3 | 0.0 | 1.2 |
| SC-err | 0.0 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 |
| GC-all | 4.3 | 28.8 | 2.2 | 39.5 | 334.3 | 1.3 |
| GC-err | 0.8 | 1.7 | 0.7 | 1.3 | 2.1 | 0.2 |

## Sweeps until no overlap was found (SC-all, GC-all)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| SC-all | 1.0 | 3.0 | 2.0 | 2.2 | 2.1 | 2.8 |
| GC-all | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |

## Training errors skipped: the overlap test found no overlap

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| SC-err | 59.6 | 271.3 | 218.5 | 19.1 | 300.0 | 63.5 |
| GC-err | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Seconds (fit for B0 and GFMM; the repair step for the others)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 | 0.2 | 0.2 | 0.1 | 0.4 | 0.3 | 4.5 |
| SR | 1.7 | 60.3 | 4.4 | 105.3 | 563.6 | 40.2 |
| SC-all | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 0.0 |
| SC-err | 0.1 | 0.4 | 0.1 | 0.6 | 2.2 | 1.0 |
| GC-all | 0.0 | 0.1 | 0.0 | 0.3 | 1.8 | 0.0 |
| GC-err | 0.1 | 0.4 | 0.1 | 0.6 | 2.2 | 1.0 |
| GFMM | 3.4 | 11.6 | 1.6 | 28.1 | 128.5 | 11.6 |

## Simpson's cases used (contractions, summed over datasets and seeds)

Cases 1 and 2: the boxes overlap partly on the contracted feature. Cases 3 and 4: one box lies inside the other there.

| Arm | Case 1 | Case 2 | Case 3 | Case 4 |
|---|---|---|---|---|
| SC-all | 5,239 | 5,039 | 521 | 1,798 |
| SC-err | 388 | 426 | 170 | 196 |

