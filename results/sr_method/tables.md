# Self-repair as the method: time, gains and RQ4 for self-repair alone

Written by `scripts/sr_method_tables.py` from saved results (RQ3: `results/refinement_v2_confirmatory/runs.csv`; Run T: `results/repair_timing/runs.csv`). The RQ3 endpoints it marks were recomputed and match RQ3's summary.md.

## Time on one machine (Run T, seed 50, seconds)

Intel64 Family 6 Model 191 Stepping 2, GenuineIntel, 16 logical CPUs, Python 3.14.10; one task at a time, no certificate audit. Every setup's test macro-F1 equals RQ3's: yes. SR+AL is SR with the cached gate, then AL (the AL stage alone in brackets). AL has no cached gate.

| Setup | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| Train CFMNN | 0.06 | 0.06 | 0.04 | 0.19 | 0.17 | 1.87 |
| SR, old gate | 0.80 | 25.63 | 1.96 | 48.57 | 210.62 | 12.49 |
| SR, cached gate | 0.20 | 1.44 | 0.82 | 2.15 | 4.68 | 2.52 |
| AL | 6.75 | 16.10 | 3.92 | 31.09 | 103.02 | 142.02 |
| SR+AL | 6.79 (6.59) | 17.07 (15.63) | 4.57 (3.74) | 33.29 (31.14) | 117.47 (112.79) | 145.04 (142.51) |

As a multiple of SR with the cached gate:

| Setup | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| SR, old gate | 4.0× | 17.8× | 2.4× | 22.6× | 45.0× | 4.9× |
| AL | 33.8× | 11.2× | 4.8× | 14.5× | 22.0× | 56.3× |
| SR+AL | 34.0× | 11.9× | 5.5× | 15.5× | 25.1× | 57.5× |

## Gains over B0 (points, mean of paired differences over seeds 50–59)

B0 is the absolute start. `*`: pre-registered two-sided test, Holm p < 0.05 (S3a for SR over B0, P3 for SR+AL over SR). `†`: pre-registered non-inferiority test passed, lowest class recall no worse than B0 within 2 points (S3d for SR, S3c for SR+AL). Other cells are descriptive.

| Macro-F1 | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 (absolute) | 83.96 | 81.98 | 67.56 | 72.59 | 74.34 | 93.10 |
| SR only | +6.02* | +1.48* | +2.58* | +3.27* | +2.25* | +5.66* |
| AL only | +6.38 | +1.94 | +4.88 | +5.06 | +1.92 | +6.13 |
| SR+AL | +6.59 | +2.56 | +4.86 | +5.30 | +2.98 | +6.21 |
| SR+AL over SR | +0.57 | +1.08* | +2.28 | +2.03* | +0.73* | +0.55 |

| Accuracy | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 (absolute) | 96.97 | 84.78 | 88.78 | 74.03 | 74.53 | 95.81 |
| SR only | +1.07 | +1.20 | +2.03 | +3.91 | +2.24 | +3.37 |
| AL only | +1.14 | +1.54 | +4.23 | +5.47 | +1.93 | +3.68 |
| SR+AL | +1.21 | +2.13 | +4.58 | +5.88 | +2.95 | +3.73 |

| Lowest class recall | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 (absolute) | 59.19 | 47.29 | 52.42 | 71.86 | 49.65 | 80.92 |
| SR only | +19.37† | +6.41† | +0.08 | −2.73 | +2.93† | +16.65† |
| AL only | +19.24 | +8.02 | −1.87 | +0.35 | +1.93 | +17.39 |
| SR+AL | +20.48† | +9.14† | −1.26 | −1.22 | +3.76† | +17.87† |

S3b (SR no worse than AL alone, within 1 point of macro-F1, one-sided): passed on 1 of 6 (Letter).

| Labels beyond the training rows | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| SR | 0 | 0 | 0 | 0 | 0 | 0 |
| AL | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |
| SR+AL | 1,020 | 944 | 821 | 2,202 | 2,300 | 24,806 |

## RQ4 for self-repair alone (post hoc)

RQ3's P4, S4a and S4b tests (paired two-sided Wilcoxon over seeds, Holm across datasets), with SR in place of SR+AL. Not pre-registered. The shuffled retrain used every label SR+AL used (training rows, bought labels, acceptance rows); SR used none. `*`: Holm p < 0.05.

| Comparison | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| Fix share: SR − retrain (points) | +34.61* | +6.61 | +26.81 | +13.16* | +15.01* | +46.97* |
| Broken test predictions: retrain − SR (points of the test rows) | +0.83* | +3.66* | +6.08* | +8.35* | +8.13* | +3.28* |
| Explanations changed (PACE-flip top 3): retrain − SR (points of the explained rows) | +36.30* | +64.50* | +58.10* | +22.90* | +52.60* | – |

Skin has 3 features, so its top-3 set never changes (excluded, as in S4b).

| Fix share (%), mean | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| SR | 86.4 | 59.0 | 69.4 | 60.3 | 65.9 | 87.5 |
| SR+AL | 85.3 | 64.2 | 81.9 | 64.8 | 66.2 | 94.1 |
| Retrain, shuffled | 51.8 | 52.4 | 42.6 | 47.2 | 50.9 | 40.5 |

Certificates on SR's accepted edits (6,233 edits, 2,000 test rows each): loose missed 0, tight missed 0 (a miss is a changed prediction the certificate did not flag).

| Certificate size (% of test rows per edit) | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| loose | 27.31 | 3.99 | 28.61 | 8.43 | 1.59 | 3.81 |
| tight | 1.23 | 2.03 | 20.04 | 7.64 | 1.42 | 3.08 |
