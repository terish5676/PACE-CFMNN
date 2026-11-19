# Run I: generality (Simpson's contraction) and sensitivity (γ)

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The paper's 80/20 split, θ = 0.3. Written by `scripts/run_generality.py`.

Seeds 150–159 (10), 13 datasets. Each comparison: paired two-sided Wilcoxon over seeds, Holm across datasets; better / level / worse for the first model.

## Primary: test accuracy at γ = 2 and γ = 8 against γ = 4

**gamma=2 vs gamma=4**: better 2, level 11, worse 0

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | +2.09 | 0.0020 | 0.0254 | better |
| S2 | +1.44 | 0.0273 | 0.2734 | level |
| S3 | +0.97 | 0.0195 | 0.2148 | level |
| wine | +0.83 | 0.2500 | 1.0000 | level |
| breast_cancer | -0.18 | 0.7500 | 1.0000 | level |
| digits | +0.97 | 0.0039 | 0.0469 | better |
| thyroid | -0.03 | 0.8047 | 1.0000 | level |
| shuttle | -0.00 | 0.5000 | 1.0000 | level |
| skin | -0.03 | 0.9219 | 1.0000 | level |
| magic | +0.33 | 0.7695 | 1.0000 | level |
| letter | +0.02 | 0.8438 | 1.0000 | level |
| landsat | +0.19 | 0.2188 | 1.0000 | level |
| page_blocks | +0.05 | 0.7500 | 1.0000 | level |

**gamma=8 vs gamma=4**: better 0, level 9, worse 4

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -4.26 | 0.0020 | 0.0254 | worse |
| S2 | -2.64 | 0.0039 | 0.0391 | worse |
| S3 | -3.85 | 0.0020 | 0.0254 | worse |
| wine | +0.28 | 0.6094 | 1.0000 | level |
| breast_cancer | +0.88 | 0.0938 | 0.8438 | level |
| digits | -0.94 | 0.0020 | 0.0254 | worse |
| thyroid | +0.03 | 0.5469 | 1.0000 | level |
| shuttle | -0.00 | 0.2500 | 1.0000 | level |
| skin | +0.56 | 0.4922 | 1.0000 | level |
| magic | +0.60 | 0.4473 | 1.0000 | level |
| letter | -0.13 | 0.4473 | 1.0000 | level |
| landsat | -0.05 | 0.4160 | 1.0000 | level |
| page_blocks | +0.13 | 0.9375 | 1.0000 | level |

## Part B: CFMNN at γ = 2, 4 and 8 (means over seeds)

Test accuracy (%):

| Dataset | gamma=2 | gamma=4 | gamma=8 |
|---|---|---|---|
| S1 | 91.80 | 89.71 | 85.45 |
| S2 | 82.02 | 80.58 | 77.94 |
| S3 | 88.86 | 87.89 | 84.04 |
| wine | 98.06 | 97.22 | 97.50 |
| breast_cancer | 95.18 | 95.35 | 96.23 |
| digits | 97.28 | 96.31 | 95.36 |
| thyroid | 96.60 | 96.62 | 96.65 |
| shuttle | 99.79 | 99.79 | 99.79 |
| skin | 94.51 | 94.54 | 95.10 |
| magic | 71.52 | 71.20 | 71.80 |
| letter | 75.38 | 75.35 | 75.22 |
| landsat | 85.84 | 85.66 | 85.61 |
| page_blocks | 87.42 | 87.37 | 87.50 |

Test macro-F1 (%):

| Dataset | gamma=2 | gamma=4 | gamma=8 |
|---|---|---|---|
| S1 | 91.80 | 89.71 | 85.44 |
| S2 | 82.01 | 80.57 | 77.92 |
| S3 | 87.75 | 86.68 | 82.46 |
| wine | 98.09 | 97.36 | 97.66 |
| breast_cancer | 94.78 | 94.97 | 95.91 |
| digits | 97.27 | 96.27 | 95.31 |
| thyroid | 82.97 | 83.01 | 82.86 |
| shuttle | 88.80 | 88.92 | 88.76 |
| skin | 90.69 | 90.76 | 91.84 |
| magic | 70.23 | 69.92 | 70.45 |
| letter | 75.16 | 75.12 | 75.00 |
| landsat | 83.23 | 82.99 | 83.01 |
| page_blocks | 63.28 | 63.18 | 64.48 |

Boxes:

| Dataset | gamma=2 | gamma=4 | gamma=8 |
|---|---|---|---|
| S1 | 1,504.9 | 1,517.7 | 1,525.7 |
| S2 | 1,514.9 | 1,533.1 | 1,544.9 |
| S3 | 1,522.0 | 1,528.7 | 1,538.7 |
| wine | 31.5 | 31.5 | 31.8 |
| breast_cancer | 42.8 | 43.0 | 43.0 |
| digits | 291.6 | 292.6 | 294.8 |
| thyroid | 51.6 | 52.0 | 52.3 |
| shuttle | 14.4 | 14.4 | 14.4 |
| skin | 88.1 | 86.2 | 88.8 |
| magic | 287.4 | 285.3 | 286.6 |
| letter | 578.6 | 576.3 | 579.0 |
| landsat | 200.3 | 200.1 | 199.5 |
| page_blocks | 24.6 | 24.6 | 24.8 |

Training error (%):

| Dataset | gamma=2 | gamma=4 | gamma=8 |
|---|---|---|---|
| S1 | 0.00 | 0.00 | 0.00 |
| S2 | 0.01 | 0.00 | 0.01 |
| S3 | 0.00 | 0.00 | 0.00 |
| wine | 0.00 | 0.00 | 0.00 |
| breast_cancer | 0.88 | 0.79 | 0.86 |
| digits | 0.00 | 0.00 | 0.00 |
| thyroid | 2.30 | 2.18 | 2.39 |
| shuttle | 0.19 | 0.19 | 0.19 |
| skin | 5.49 | 5.42 | 4.93 |
| magic | 26.59 | 26.49 | 25.68 |
| letter | 20.92 | 20.92 | 20.93 |
| landsat | 10.49 | 10.50 | 10.52 |
| page_blocks | 11.78 | 11.88 | 11.78 |

Share of training errors that are ties (%), all seeds pooled; – = no training error. Proposition 1 predicts 100 at every γ:

| Dataset | gamma=2 | gamma=4 | gamma=8 |
|---|---|---|---|
| S1 | – | 100.0 | – |
| S2 | 100.0 | 100.0 | 100.0 |
| S3 | 100.0 | 100.0 | – |
| wine | – | – | – |
| breast_cancer | 100.0 | 100.0 | 100.0 |
| digits | – | – | – |
| thyroid | 100.0 | 100.0 | 100.0 |
| shuttle | 100.0 | 100.0 | 100.0 |
| skin | 100.0 | 100.0 | 100.0 |
| magic | 100.0 | 100.0 | 100.0 |
| letter | 100.0 | 100.0 | 100.0 |
| landsat | 100.0 | 100.0 | 100.0 |
| page_blocks | 100.0 | 100.0 | 100.0 |

Training errors that are not ties, over every dataset, seed and γ: **0** of 569,239.

Share of test predictions decided by the nearest centroid (%), all seeds pooled:

| Dataset | gamma=2 | gamma=4 | gamma=8 |
|---|---|---|---|
| S1 | 0.0 | 0.0 | 0.1 |
| S2 | 0.0 | 0.0 | 0.1 |
| S3 | 0.0 | 0.0 | 0.1 |
| wine | 0.0 | 0.0 | 0.0 |
| breast_cancer | 0.4 | 0.3 | 0.3 |
| digits | 0.1 | 0.2 | 0.8 |
| thyroid | 5.5 | 5.5 | 5.6 |
| shuttle | 13.6 | 13.6 | 13.6 |
| skin | 20.7 | 20.3 | 19.9 |
| magic | 77.7 | 77.5 | 77.7 |
| letter | 55.7 | 55.6 | 55.7 |
| landsat | 25.3 | 25.2 | 25.1 |
| page_blocks | 68.1 | 67.9 | 67.8 |

Share of test errors that are among those predictions (%), all seeds pooled:

| Dataset | gamma=2 | gamma=4 | gamma=8 |
|---|---|---|---|
| S1 | 0.0 | 0.0 | 0.1 |
| S2 | 0.0 | 0.0 | 0.2 |
| S3 | 0.0 | 0.0 | 0.6 |
| wine | 0.0 | 0.0 | 0.0 |
| breast_cancer | 7.3 | 5.7 | 7.0 |
| digits | 2.0 | 3.8 | 6.6 |
| thyroid | 65.5 | 64.8 | 67.4 |
| shuttle | 83.4 | 83.7 | 83.1 |
| skin | 99.8 | 99.8 | 99.8 |
| magic | 90.1 | 89.8 | 89.9 |
| letter | 90.4 | 90.2 | 89.9 |
| landsat | 57.4 | 57.2 | 57.0 |
| page_blocks | 95.2 | 95.1 | 95.2 |

## Part A (2): Simpson's contraction next to CFMNN (γ = 4)

What the contraction did (means over seeds):

| Dataset | Boxes before | Contractions | Seeds with no contraction | Boxes removed | Training rows evicted | Boxes after |
|---|---|---|---|---|---|---|
| S1 | 1,517.7 | 56.9 | 0 of 10 | 1.7 | 110.4 | 1,516.0 |
| S2 | 1,533.1 | 151.8 | 0 of 10 | 10.9 | 281.2 | 1,522.2 |
| S3 | 1,528.7 | 75.7 | 0 of 10 | 5.0 | 137.2 | 1,523.7 |
| wine | 31.5 | 0.1 | 9 of 10 | 0.0 | 0.2 | 31.5 |
| breast_cancer | 43.0 | 35.8 | 0 of 10 | 0.2 | 91.3 | 42.8 |
| digits | 292.6 | 0.0 | 10 of 10 | 0.0 | 0.0 | 292.6 |
| thyroid | 52.0 | 0.0 | 10 of 10 | 0.0 | 0.0 | 52.0 |
| shuttle | 14.4 | 4.7 | 0 of 10 | 0.4 | 40,450.0 | 14.0 |
| skin | 86.2 | 50.0 | 0 of 10 | 0.5 | 30,498.2 | 85.7 |
| magic | 285.3 | 1,559.1 | 0 of 10 | 43.8 | 13,788.8 | 241.5 |
| letter | 576.3 | 20.1 | 0 of 10 | 0.0 | 232.3 | 576.3 |
| landsat | 200.1 | 143.1 | 0 of 10 | 1.6 | 976.7 | 198.5 |
| page_blocks | 24.6 | 12.8 | 0 of 10 | 0.5 | 688.2 | 24.1 |

Training error (%):

| Dataset | gamma=4 | Simpson |
|---|---|---|
| S1 | 0.00 | 0.03 |
| S2 | 0.00 | 0.30 |
| S3 | 0.00 | 0.07 |
| wine | 0.00 | 0.00 |
| breast_cancer | 0.79 | 1.21 |
| digits | 0.00 | 0.00 |
| thyroid | 2.18 | 2.18 |
| shuttle | 0.19 | 28.63 |
| skin | 5.42 | 11.04 |
| magic | 26.49 | 40.91 |
| letter | 20.92 | 20.91 |
| landsat | 10.50 | 11.06 |
| page_blocks | 11.88 | 19.07 |

Share of training errors that are ties (%), all seeds pooled; – = no training error. A tie: the row's own class reaches the winner's membership:

| Dataset | gamma=4 | Simpson |
|---|---|---|
| S1 | 100.0 | 0.0 |
| S2 | 100.0 | 0.0 |
| S3 | 100.0 | 0.0 |
| wine | – | – |
| breast_cancer | 100.0 | 3.6 |
| digits | – | – |
| thyroid | 100.0 | 100.0 |
| shuttle | 100.0 | 0.1 |
| skin | 100.0 | 7.7 |
| magic | 100.0 | 0.2 |
| letter | 100.0 | 98.6 |
| landsat | 100.0 | 70.4 |
| page_blocks | 100.0 | 63.9 |

... of which inside a box of both classes (both memberships are 1), the form Proposition 1 gives for CFMNN (%):

| Dataset | gamma=4 | Simpson |
|---|---|---|
| S1 | 100.0 | 0.0 |
| S2 | 100.0 | 0.0 |
| S3 | 100.0 | 0.0 |
| wine | – | – |
| breast_cancer | 100.0 | 3.6 |
| digits | – | – |
| thyroid | 100.0 | 100.0 |
| shuttle | 100.0 | 0.1 |
| skin | 100.0 | 7.7 |
| magic | 100.0 | 0.2 |
| letter | 100.0 | 98.6 |
| landsat | 100.0 | 70.3 |
| page_blocks | 100.0 | 63.8 |

Test accuracy (%):

| Dataset | gamma=4 | Simpson |
|---|---|---|
| S1 | 89.71 | 89.70 |
| S2 | 80.58 | 80.73 |
| S3 | 87.89 | 87.97 |
| wine | 97.22 | 97.22 |
| breast_cancer | 95.35 | 95.53 |
| digits | 96.31 | 96.31 |
| thyroid | 96.62 | 96.62 |
| shuttle | 99.79 | 71.18 |
| skin | 94.54 | 88.96 |
| magic | 71.20 | 58.33 |
| letter | 75.35 | 75.21 |
| landsat | 85.66 | 85.12 |
| page_blocks | 87.37 | 80.46 |

Test macro-F1 (%):

| Dataset | gamma=4 | Simpson |
|---|---|---|
| S1 | 89.71 | 89.70 |
| S2 | 80.57 | 80.72 |
| S3 | 86.68 | 86.77 |
| wine | 97.36 | 97.36 |
| breast_cancer | 94.97 | 95.15 |
| digits | 96.27 | 96.27 |
| thyroid | 83.01 | 83.01 |
| shuttle | 88.92 | 63.87 |
| skin | 90.76 | 77.82 |
| magic | 69.92 | 57.66 |
| letter | 75.12 | 74.98 |
| landsat | 82.99 | 82.40 |
| page_blocks | 63.18 | 58.02 |

Share of test predictions decided by the nearest centroid (%), all seeds pooled:

| Dataset | gamma=4 | Simpson |
|---|---|---|
| S1 | 0.0 | 0.0 |
| S2 | 0.0 | 0.0 |
| S3 | 0.0 | 0.0 |
| wine | 0.0 | 0.0 |
| breast_cancer | 0.3 | 0.0 |
| digits | 0.2 | 0.2 |
| thyroid | 5.5 | 5.5 |
| shuttle | 13.6 | 1.3 |
| skin | 20.3 | 5.5 |
| magic | 77.5 | 0.2 |
| letter | 55.6 | 55.3 |
| landsat | 25.2 | 20.2 |
| page_blocks | 67.9 | 54.5 |

## Part A (1): PACE-SHAP on the Simpson model against shap's ExactExplainer

**Check: PASS** (largest difference 1.6e-10; the plan's limit is 1e-09).

| Dataset | Features | Seeds | Points | Contractions (mean) | Seeds with no contraction | Boxes (mean) | Largest difference | Largest value | ms PACE-SHAP | ms ExactExplainer |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 10 | 10 | 50 | 56.9 | 0 of 10 | 1,516.0 | 1.3e-10 | 0.085 | 7,886.6 | 81,834.0 |
| S2 | 10 | 10 | 50 | 151.8 | 0 of 10 | 1,522.2 | 1.1e-10 | 0.078 | 6,846.0 | 81,294.9 |
| S3 | 10 | 10 | 50 | 75.7 | 0 of 10 | 1,523.7 | 1.6e-10 | 0.110 | 5,607.1 | 63,363.8 |
| shuttle | 9 | 10 | 50 | 4.7 | 0 of 10 | 14.0 | 2.7e-16 | 0.176 | 19.0 | 412.7 |
| magic | 10 | 10 | 50 | 1,559.1 | 0 of 10 | 241.5 | 7.1e-11 | 0.076 | 579.8 | 14,232.5 |
| page_blocks | 10 | 10 | 50 | 12.8 | 0 of 10 | 24.1 | 1.8e-16 | 0.032 | 65.1 | 1,740.0 |

Largest difference: the largest absolute difference between the two, over every point, feature and seed. Where a seed has no contraction, the Simpson model is the CFMNN itself.

