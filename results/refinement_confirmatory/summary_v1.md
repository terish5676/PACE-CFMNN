# Refinement v1: summary of the pre-registered run

Recomputed by `scripts/summarise_refinement_v1.py` from `runs.csv` and `rounds.csv` (copied unchanged from the teammate's run). Seeds 10–19, 6 datasets, 10 seeds each. Our own code reproduces the pre-registered arms exactly (Page Blocks, seed 10, all 7 arms).

Metric: test macro-F1 (%) after the last round. Test: paired two-sided Wilcoxon over the 10 seeds, Holm across the 6 datasets of each comparison. `*` = Holm-adjusted p < 0.05. Differences are in macro-F1 points, mean over seeds.

## 1. Pre-registered endpoints

| Endpoint | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| R1 (primary): A+B+C − A+B+C′-volume | -1.98 | +0.22 | +2.00 | -0.88 | -0.88 | -0.34 |
| R2: A+B+C − A+B | +4.23* | +1.18* | +8.61* | +5.30* | +0.74* | +6.49* |
| R3: A+B+C − A+B+C′-random | -0.42 | -0.25 | +0.85 | -0.34 | -0.86* | -0.16 |
| S1: order ensemble − random selection (both A+B+C) | +2.13 | +0.77 | +6.50* | +2.53* | +0.57 | +1.39* |

Wins–losses over the 10 seeds, raw p and Holm-adjusted p:

| Endpoint | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| R1 (primary) | 2–8, p=0.064, Holm 0.258 | 6–4, p=0.375, Holm 0.551 | 7–3, p=0.275, Holm 0.551 | 3–7, p=0.131, Holm 0.393 | 3–7, p=0.027, Holm 0.137 | 2–8, p=0.010, Holm 0.059 |
| R2 | 8–2, p=0.014, Holm 0.027 | 8–2, p=0.027, Holm 0.027 | 10–0, p=0.002, Holm 0.012 | 10–0, p=0.002, Holm 0.012 | 10–0, p=0.002, Holm 0.012 | 9–1, p=0.004, Holm 0.012 |
| R3 | 5–5, p=0.695, Holm 1.000 | 4–6, p=0.432, Holm 1.000 | 5–5, p=0.695, Holm 1.000 | 5–5, p=0.432, Holm 1.000 | 0–10, p=0.002, Holm 0.012 | 3–7, p=0.375, Holm 1.000 |
| S1 | 6–4, p=0.193, Holm 0.252 | 8–2, p=0.105, Holm 0.252 | 9–1, p=0.004, Holm 0.020 | 10–0, p=0.002, Holm 0.012 | 6–4, p=0.084, Holm 0.252 | 9–1, p=0.006, Holm 0.023 |

**Reading:**

- **R1 (primary, RQ4):** the explanation's cut is not better than the smallest-cut rule on any dataset (0 of 6 significant; mean difference -1.98 to +2.00). The blueprint hypothesis is not supported.
- **R2:** shrinking behind the gate beats add-only on 6 of 6 datasets (+0.74 to +8.61 points).
- **S1:** order-ensemble selection beats random selection on 3 of 6 datasets.

## 2. Descriptive measures (pre-registered arms)

### Test macro-F1 (%), mean over 10 seeds

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 (no refinement) | 85.3 | 81.3 | 65.1 | 71.1 | 74.3 | 92.2 |
| B9 (refit on all pool labels) | 85.4 | 81.5 | 65.4 | 71.0 | 75.6 | 92.1 |
| OE, A+B | 86.2 | 82.0 | 66.0 | 71.1 | 74.9 | 92.4 |
| OE, A+B+C (explanation cut) | 90.5 | 83.2 | 74.6 | 76.4 | 75.7 | 98.9 |
| OE, A+B+C′-random | 90.9 | 83.5 | 73.7 | 76.7 | 76.5 | 99.1 |
| OE, A+B+C′-volume | 92.4 | 83.0 | 72.6 | 77.2 | 76.6 | 99.2 |
| Random, A+B+C | 88.3 | 82.5 | 68.1 | 73.8 | 75.1 | 97.5 |

### Lowest class recall (%), mean over 10 seeds

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 (no refinement) | 60.7 | 46.9 | 46.9 | 70.3 | 46.1 | 79.3 |
| B9 (refit on all pool labels) | 59.9 | 47.2 | 50.6 | 70.8 | 48.8 | 78.9 |
| OE, A+B | 62.6 | 47.2 | 50.2 | 70.3 | 47.2 | 79.6 |
| OE, A+B+C (explanation cut) | 77.7 | 53.1 | 54.4 | 70.0 | 47.7 | 97.7 |
| OE, A+B+C′-random | 84.5 | 57.8 | 47.7 | 71.5 | 48.6 | 98.4 |
| OE, A+B+C′-volume | 81.9 | 54.2 | 45.4 | 71.7 | 49.6 | 98.7 |
| Random, A+B+C | 70.3 | 51.9 | 48.1 | 70.1 | 46.5 | 94.0 |

### Test accuracy (%), mean over 10 seeds

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 (no refinement) | 97.0 | 84.2 | 86.6 | 72.4 | 74.5 | 95.4 |
| B9 (refit on all pool labels) | 97.1 | 84.3 | 84.9 | 72.3 | 75.8 | 95.3 |
| OE, A+B | 97.2 | 84.9 | 86.5 | 72.3 | 75.2 | 95.5 |
| OE, A+B+C (explanation cut) | 97.9 | 85.8 | 92.5 | 78.4 | 75.9 | 99.3 |
| OE, A+B+C′-random | 98.0 | 85.8 | 93.0 | 78.6 | 76.8 | 99.4 |
| OE, A+B+C′-volume | 98.4 | 85.5 | 92.3 | 79.1 | 76.8 | 99.5 |
| Random, A+B+C | 97.5 | 85.0 | 89.1 | 75.7 | 75.4 | 98.4 |

### Boxes at the end (mean)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 (no refinement) | 44 | 153 | 23 | 233 | 456 | 80 |
| B9 (refit on all pool labels) | 51 | 196 | 26 | 288 | 539 | 84 |
| OE, A+B | 47 | 165 | 24 | 238 | 468 | 80 |
| OE, A+B+C (explanation cut) | 44 | 163 | 24 | 232 | 466 | 80 |
| OE, A+B+C′-random | 42 | 163 | 24 | 233 | 466 | 80 |
| OE, A+B+C′-volume | 46 | 164 | 24 | 236 | 467 | 80 |
| Random, A+B+C | 44 | 163 | 23 | 237 | 462 | 80 |

### Share of picked points the model got wrong (%)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE, A+B | 15.5 | 52.2 | 47.1 | 77.7 | 90.0 | 91.8 |
| OE, A+B+C (explanation cut) | 13.1 | 50.8 | 35.3 | 63.1 | 88.6 | 35.9 |
| OE, A+B+C′-random | 12.8 | 49.4 | 29.7 | 63.4 | 86.3 | 30.5 |
| OE, A+B+C′-volume | 10.5 | 49.5 | 31.4 | 64.6 | 88.1 | 27.9 |
| Random, A+B+C | 2.9 | 14.8 | 11.8 | 25.5 | 25.1 | 2.5 |

### Shrinks tried (mean per run)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE, A+B+C (explanation cut) | 23 | 104 | 88 | 165 | 232 | 102 |
| OE, A+B+C′-random | 19 | 100 | 74 | 163 | 224 | 84 |
| OE, A+B+C′-volume | 19 | 103 | 79 | 173 | 230 | 78 |
| Random, A+B+C | 7 | 30 | 30 | 72 | 69 | 7 |

### Shrinks accepted by the gate (% of tried)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE, A+B+C (explanation cut) | 62 | 51 | 15 | 94 | 43 | 59 |
| OE, A+B+C′-random | 68 | 49 | 17 | 95 | 46 | 74 |
| OE, A+B+C′-volume | 75 | 67 | 20 | 99 | 55 | 85 |
| Random, A+B+C | 78 | 63 | 18 | 88 | 47 | 77 |

### Mean share of the box's width cut by an accepted shrink

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE, A+B+C (explanation cut) | 0.225 | 0.300 | 0.242 | 0.286 | 0.239 | 0.234 |
| OE, A+B+C′-random | 0.621 | 0.329 | 0.173 | 0.283 | 0.248 | 0.220 |
| OE, A+B+C′-volume | 0.001 | 0.056 | 0.017 | 0.068 | 0.005 | 0.097 |
| Random, A+B+C | 0.323 | 0.288 | 0.242 | 0.301 | 0.241 | 0.286 |

### Mean training members cut per accepted shrink

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| OE, A+B+C (explanation cut) | 130 | 10 | 71 | 29 | 11 | 589 |
| OE, A+B+C′-random | 160 | 12 | 125 | 31 | 10 | 655 |
| OE, A+B+C′-volume | 121 | 4 | 41 | 10 | 5 | 175 |
| Random, A+B+C | 165 | 13 | 92 | 53 | 13 | 412 |

### Learning curve: test macro-F1 (%) at rounds 0, 5 and 10

| Arm | Round | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|---|
| OE, A+B | 0 | 85.3 | 81.3 | 65.1 | 71.1 | 74.3 | 92.2 |
| OE, A+B | 5 | 85.8 | 81.7 | 65.4 | 71.0 | 74.6 | 92.4 |
| OE, A+B | 10 | 86.2 | 82.0 | 66.0 | 71.1 | 74.9 | 92.4 |
| OE, A+B+C (explanation cut) | 0 | 85.3 | 81.3 | 65.1 | 71.1 | 74.3 | 92.2 |
| OE, A+B+C (explanation cut) | 5 | 90.5 | 82.8 | 71.7 | 75.7 | 74.9 | 98.7 |
| OE, A+B+C (explanation cut) | 10 | 90.5 | 83.2 | 74.6 | 76.4 | 75.7 | 98.9 |
| OE, A+B+C′-random | 0 | 85.3 | 81.3 | 65.1 | 71.1 | 74.3 | 92.2 |
| OE, A+B+C′-random | 5 | 90.6 | 83.5 | 73.1 | 75.4 | 75.7 | 98.9 |
| OE, A+B+C′-random | 10 | 90.9 | 83.5 | 73.7 | 76.7 | 76.5 | 99.1 |
| OE, A+B+C′-volume | 0 | 85.3 | 81.3 | 65.1 | 71.1 | 74.3 | 92.2 |
| OE, A+B+C′-volume | 5 | 92.5 | 82.6 | 70.9 | 75.2 | 75.6 | 98.8 |
| OE, A+B+C′-volume | 10 | 92.4 | 83.0 | 72.6 | 77.2 | 76.6 | 99.2 |
| Random, A+B+C | 0 | 85.3 | 81.3 | 65.1 | 71.1 | 74.3 | 92.2 |
| Random, A+B+C | 5 | 86.8 | 82.1 | 67.1 | 73.3 | 74.7 | 97.0 |
| Random, A+B+C | 10 | 88.3 | 82.5 | 68.1 | 73.8 | 75.1 | 97.5 |

### Run time (seconds per run, mean)

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B0 (no refinement) | 0 | 0 | 0 | 0 | 0 | 0 |
| B9 (refit on all pool labels) | 0 | 0 | 0 | 0 | 1 | 3 |
| OE, A+B | 11 | 23 | 6 | 32 | 101 | 177 |
| OE, A+B+C (explanation cut) | 11 | 24 | 7 | 41 | 130 | 166 |
| OE, A+B+C′-random | 11 | 26 | 6 | 46 | 124 | 168 |
| OE, A+B+C′-volume | 12 | 24 | 6 | 48 | 126 | 181 |
| Random, A+B+C | 5 | 17 | 2 | 14 | 70 | 44 |

## 3. Every label each arm used

The gate reads the labelled acceptance rows for every edit. They are labels too, and the pre-registered budget of 300 did not count them. B0 and B9 read none of them.

| Dataset | Arm | Pool labels | Acceptance labels | Total |
|---|---|---|---|---|
| Thyroid | B0 (no refinement) | 0 | 0 | 0 |
| Thyroid | B9 (refit on all pool labels) | 1,440 | 0 | 1,440 |
| Thyroid | B-Simpson (amended) | 0 | 0 | 0 |
| Thyroid | gated arms (all OE and random arms) | 300 | 720 | 1,020 |
| Landsat | B0 (no refinement) | 0 | 0 | 0 |
| Landsat | B9 (refit on all pool labels) | 1,287 | 0 | 1,287 |
| Landsat | B-Simpson (amended) | 0 | 0 | 0 |
| Landsat | gated arms (all OE and random arms) | 300 | 644 | 944 |
| Page Blocks | B0 (no refinement) | 0 | 0 | 0 |
| Page Blocks | B9 (refit on all pool labels) | 1,094 | 0 | 1,094 |
| Page Blocks | B-Simpson (amended) | 0 | 0 | 0 |
| Page Blocks | gated arms (all OE and random arms) | 273 | 548 | 821 |
| MAGIC | B0 (no refinement) | 0 | 0 | 0 |
| MAGIC | B9 (refit on all pool labels) | 3,804 | 0 | 3,804 |
| MAGIC | B-Simpson (amended) | 0 | 0 | 0 |
| MAGIC | gated arms (all OE and random arms) | 300 | 1,902 | 2,202 |
| Letter | B0 (no refinement) | 0 | 0 | 0 |
| Letter | B9 (refit on all pool labels) | 4,000 | 0 | 4,000 |
| Letter | B-Simpson (amended) | 0 | 0 | 0 |
| Letter | gated arms (all OE and random arms) | 300 | 2,000 | 2,300 |
| Skin | B0 (no refinement) | 0 | 0 | 0 |
| Skin | B9 (refit on all pool labels) | 49,011 | 0 | 49,011 |
| Skin | B-Simpson (amended) | 0 | 0 | 0 |
| Skin | gated arms (all OE and random arms) | 300 | 24,506 | 24,806 |

The gated arms used 2.0–82× as many acceptance labels as pool labels.

## Appendix: the two arms added by amendment (not pre-registered)

Reported for completeness. Neither is used in the v2 proposal.

- **C-minimal** keeps the features whose contrastive term favours the *wrong* class (the amendment describes it the other way round) and cuts the thinnest of them. It picks the same feature as C′-volume on 7–70% of eligible points, so R4 compares a half-geometric rule with the geometric rule. Its comparison with C′-volume was designed after the pilot had shown C′-volume winning, so it cannot replace R1 as the primary endpoint.
- **B-Simpson** contracts overlapping boxes once after training. It finds no overlaps on Thyroid (zero-width sides on binary features are treated as open), so there it equals B0 exactly; elsewhere it drops thousands of training members and deletes boxes, without regrowing anything. It has no gate and uses no labels. It is not a fair stand-in for Simpson's contraction, so R6 and the Simpson rows say nothing about Simpson's method.

| Endpoint | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| R4: C-minimal − C′-volume | -2.05 | +0.50 | -1.40 | -1.11* | -0.60 | -0.42* |
| R5: C-minimal − C | -0.07 | +0.28 | -3.39* | -0.23 | +0.28 | -0.08 |
| R6: C-minimal − B-Simpson | +5.07* | +3.29* | +18.05* | +14.37* | +14.91* | +20.22* |
| extra: B-Simpson − B0 | +0.00 | -1.10 | -11.92* | -9.35* | -13.25* | -13.63* |

Wins–losses over the 10 seeds, raw p and Holm-adjusted p:

| Endpoint | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| R4 | 1–9, p=0.014, Holm 0.055 | 7–3, p=0.131, Holm 0.262 | 4–6, p=0.492, Holm 0.492 | 0–10, p=0.002, Holm 0.012 | 2–8, p=0.084, Holm 0.252 | 1–9, p=0.004, Holm 0.020 |
| R5 | 4–6, p=0.695, Holm 1.000 | 4–6, p=0.846, Holm 1.000 | 1–9, p=0.004, Holm 0.023 | 4–6, p=0.432, Holm 1.000 | 7–3, p=0.232, Holm 1.000 | 5–5, p=0.846, Holm 1.000 |
| R6 | 9–1, p=0.004, Holm 0.012 | 10–0, p=0.002, Holm 0.012 | 10–0, p=0.002, Holm 0.012 | 10–0, p=0.002, Holm 0.012 | 10–0, p=0.002, Holm 0.012 | 10–0, p=0.002, Holm 0.012 |
| extra | 0–0, p=1.000, Holm 1.000 | 3–7, p=0.322, Holm 0.645 | 1–9, p=0.004, Holm 0.012 | 0–10, p=0.002, Holm 0.012 | 0–10, p=0.002, Holm 0.012 | 0–10, p=0.002, Holm 0.012 |

### Test macro-F1 / lowest class recall (%), mean over 10 seeds

| Arm | Thyroid | Landsat | Page Blocks | MAGIC | Letter | Skin |
|---|---|---|---|---|---|---|
| B-Simpson (amended) | 85.3 / 60.7 | 80.2 / 33.9 | 53.2 / 37.6 | 61.8 / 56.7 | 61.1 / 14.2 | 78.6 / 48.3 |
| OE, A+B+C-minimal (amended) | 90.4 / 76.5 | 83.5 / 56.5 | 71.2 / 47.1 | 76.1 / 70.9 | 76.0 / 49.3 | 98.8 / 98.0 |
