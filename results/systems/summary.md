# Run G: CFMNN + PACE-SHAP against other models that explain themselves

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Seeds 120–129 (10). Each system is tested on its own predicted-class margin. Lead = how far the system's ranking beats a random ranking on the same model, as a share of the random ranking's AUC (larger is better). Each comparison: paired two-sided Wilcoxon over seeds, Holm across datasets; better / level / worse for CFMNN + PACE-SHAP.

## Normalised deletion lead (mean over seeds)

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Tree + TreeSHAP |
|---|---|---|---|
| S1 | 1.515 | 1.325 | 1.239 |
| S2 | 1.614 | 1.386 | 1.293 |
| S3 | 1.266 | 1.161 | 1.171 |
| wine | 1.207 | 1.244 | 1.861 |
| breast_cancer | 0.698 | 0.919 | 0.864 |
| thyroid | 0.280 | 0.267 | 0.211 |
| shuttle | 0.304 | 0.516 | 0.435 |
| magic | 1.334 | 1.454 | 1.340 |
| letter | 0.795 | 1.026 | 1.037 |
| landsat | 58.899 | 12.736 | 623.112 |
| page_blocks | 1.032 | 0.241 | 0.196 |

## Primary: normalised deletion lead

**CFMNN + PACE-SHAP vs EBM + terms**, per dataset (deletion_lead): better 2, level 6, worse 3

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 1.515 | 1.325 | +0.190 | 0.0020 | 0.0215 | better |
| S2 | 1.614 | 1.386 | +0.228 | 0.0137 | 0.0820 | level |
| S3 | 1.266 | 1.161 | +0.105 | 0.1055 | 0.5273 | level |
| wine | 1.207 | 1.244 | -0.037 | 0.3223 | 0.6973 | level |
| breast_cancer | 0.698 | 0.919 | -0.221 | 0.0020 | 0.0215 | worse |
| thyroid | 0.280 | 0.267 | +0.013 | 0.4922 | 0.6973 | level |
| shuttle | 0.304 | 0.516 | -0.212 | 0.0020 | 0.0215 | worse |
| magic | 1.334 | 1.454 | -0.120 | 0.1602 | 0.6406 | level |
| letter | 0.795 | 1.026 | -0.231 | 0.0020 | 0.0215 | worse |
| landsat | 58.899 | 12.736 | +46.163 | 0.2324 | 0.6973 | level |
| page_blocks | 1.032 | 0.241 | +0.791 | 0.0020 | 0.0215 | better |

**CFMNN + PACE-SHAP vs EBM + terms**, synthetic family as one unit (deletion_lead): better 2, level 4, worse 3

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 1.465 | 1.291 | +0.175 | 0.0020 | 0.0176 | better |
| wine | 1.207 | 1.244 | -0.037 | 0.3223 | 0.6973 | level |
| breast_cancer | 0.698 | 0.919 | -0.221 | 0.0020 | 0.0176 | worse |
| thyroid | 0.280 | 0.267 | +0.013 | 0.4922 | 0.6973 | level |
| shuttle | 0.304 | 0.516 | -0.212 | 0.0020 | 0.0176 | worse |
| magic | 1.334 | 1.454 | -0.120 | 0.1602 | 0.6406 | level |
| letter | 0.795 | 1.026 | -0.231 | 0.0020 | 0.0176 | worse |
| landsat | 58.899 | 12.736 | +46.163 | 0.2324 | 0.6973 | level |
| page_blocks | 1.032 | 0.241 | +0.791 | 0.0020 | 0.0176 | better |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, per dataset (deletion_lead): better 3, level 4, worse 4

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 1.515 | 1.239 | +0.276 | 0.0020 | 0.0215 | better |
| S2 | 1.614 | 1.293 | +0.321 | 0.0059 | 0.0293 | better |
| S3 | 1.266 | 1.171 | +0.095 | 0.1055 | 0.3164 | level |
| wine | 1.207 | 1.861 | -0.653 | 0.0020 | 0.0215 | worse |
| breast_cancer | 0.698 | 0.864 | -0.166 | 0.0020 | 0.0215 | worse |
| thyroid | 0.280 | 0.211 | +0.069 | 0.0137 | 0.0547 | level |
| shuttle | 0.304 | 0.435 | -0.131 | 0.0020 | 0.0215 | worse |
| magic | 1.334 | 1.340 | -0.006 | 0.4922 | 0.9844 | level |
| letter | 0.795 | 1.037 | -0.242 | 0.0020 | 0.0215 | worse |
| landsat | 58.899 | 623.112 | -564.213 | 0.6250 | 0.9844 | level |
| page_blocks | 1.032 | 0.196 | +0.836 | 0.0020 | 0.0215 | better |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, synthetic family as one unit (deletion_lead): better 3, level 2, worse 4

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 1.465 | 1.234 | +0.231 | 0.0020 | 0.0176 | better |
| wine | 1.207 | 1.861 | -0.653 | 0.0020 | 0.0176 | worse |
| breast_cancer | 0.698 | 0.864 | -0.166 | 0.0020 | 0.0176 | worse |
| thyroid | 0.280 | 0.211 | +0.069 | 0.0137 | 0.0410 | better |
| shuttle | 0.304 | 0.435 | -0.131 | 0.0020 | 0.0176 | worse |
| magic | 1.334 | 1.340 | -0.006 | 0.4922 | 0.9844 | level |
| letter | 0.795 | 1.037 | -0.242 | 0.0020 | 0.0176 | worse |
| landsat | 58.899 | 623.112 | -564.213 | 0.6250 | 0.9844 | level |
| page_blocks | 1.032 | 0.196 | +0.836 | 0.0020 | 0.0176 | better |

## Primary with the near-zero rule

Where any seed of a compared pair has a random deletion AUC within 0.05 of zero, the verdict is read from the deletion gain (pre-registered).

**CFMNN + PACE-SHAP vs EBM + terms**, per dataset: better 2, level 5, worse 4; read from the gain: landsat

**CFMNN + PACE-SHAP vs EBM + terms**, synthetic family as one unit: better 2, level 3, worse 4; read from the gain: landsat

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, per dataset: better 3, level 3, worse 5; read from the gain: landsat

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, synthetic family as one unit: better 3, level 1, worse 5; read from the gain: landsat

## Amendment 1: the primary on untied points

CFMNN can tie (a zero margin); for a tied point the deletion curve is divided by almost zero and becomes signs, which the other two models' points do not do. Below, every system is scored on its untied explained points only. Tested on the 9 dataset(s) where every seed of every system has at least 10 untied points: S1, S2, S3, wine, breast_cancer, thyroid, shuttle, letter, landsat.

**CFMNN + PACE-SHAP vs EBM + terms**: how the primary verdict reads (tie-heavy = CFMNN's mean share of tied points is at least 10%)

| Dataset | CFMNN's tied share | Primary (all points) | Untied points | Reading |
|---|---|---|---|---|
| S1 | 0.0% | better | better | not tie-heavy |
| S2 | 0.0% | level | level | not tie-heavy |
| S3 | 0.0% | level | level | not tie-heavy |
| wine | 0.0% | level | level | not tie-heavy |
| breast_cancer | 0.2% | worse | worse | not tie-heavy |
| thyroid | 6.2% | level | level | not tie-heavy |
| shuttle | 9.4% | worse | worse | not tie-heavy |
| magic | 77.0% | level | not tested | not checked: too few untied points |
| letter | 53.6% | worse | better | depends on the tied points |
| landsat | 25.8% | level | level | holds on untied points |
| page_blocks | 74.6% | better | not tested | not checked: too few untied points |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**: how the primary verdict reads (tie-heavy = CFMNN's mean share of tied points is at least 10%)

| Dataset | CFMNN's tied share | Primary (all points) | Untied points | Reading |
|---|---|---|---|---|
| S1 | 0.0% | better | better | not tie-heavy |
| S2 | 0.0% | better | better | not tie-heavy |
| S3 | 0.0% | level | level | not tie-heavy |
| wine | 0.0% | worse | worse | not tie-heavy |
| breast_cancer | 0.2% | worse | worse | not tie-heavy |
| thyroid | 6.2% | level | level | not tie-heavy |
| shuttle | 9.4% | worse | worse | not tie-heavy |
| magic | 77.0% | level | not tested | not checked: too few untied points |
| letter | 53.6% | worse | better | depends on the tied points |
| landsat | 25.8% | level | level | holds on untied points |
| page_blocks | 74.6% | better | not tested | not checked: too few untied points |

## Amendment 1: normalised deletion lead, untied points

**CFMNN + PACE-SHAP vs EBM + terms** (deletion_lead_untied): better 2, level 5, worse 2

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 1.515 | 1.325 | +0.190 | 0.0020 | 0.0176 | better |
| S2 | 1.614 | 1.386 | +0.228 | 0.0137 | 0.0684 | level |
| S3 | 1.266 | 1.161 | +0.105 | 0.1055 | 0.3164 | level |
| wine | 1.207 | 1.244 | -0.037 | 0.3223 | 0.6445 | level |
| breast_cancer | 0.698 | 0.919 | -0.221 | 0.0020 | 0.0176 | worse |
| thyroid | 0.245 | 0.267 | -0.023 | 0.3750 | 0.6445 | level |
| shuttle | 0.251 | 0.516 | -0.266 | 0.0020 | 0.0176 | worse |
| letter | 2.200 | 1.026 | +1.174 | 0.0020 | 0.0176 | better |
| landsat | 7.013 | 12.736 | -5.723 | 0.0645 | 0.2578 | level |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP** (deletion_lead_untied): better 3, level 3, worse 3

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 1.515 | 1.239 | +0.276 | 0.0020 | 0.0176 | better |
| S2 | 1.614 | 1.293 | +0.321 | 0.0059 | 0.0234 | better |
| S3 | 1.266 | 1.171 | +0.095 | 0.1055 | 0.2520 | level |
| wine | 1.207 | 1.861 | -0.653 | 0.0020 | 0.0176 | worse |
| breast_cancer | 0.698 | 0.864 | -0.166 | 0.0020 | 0.0176 | worse |
| thyroid | 0.245 | 0.211 | +0.034 | 0.1934 | 0.2520 | level |
| shuttle | 0.251 | 0.435 | -0.184 | 0.0020 | 0.0176 | worse |
| letter | 2.200 | 1.084 | +1.116 | 0.0020 | 0.0176 | better |
| landsat | 7.013 | 622.641 | -615.628 | 0.0840 | 0.2520 | level |

## Amendment 1: deletion gain, untied points (not divided)

**CFMNN + PACE-SHAP vs EBM + terms** (deletion_gain_untied): better 1, level 1, worse 7

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 0.427 | 0.524 | -0.097 | 0.0020 | 0.0176 | worse |
| S2 | 0.394 | 0.536 | -0.142 | 0.0020 | 0.0176 | worse |
| S3 | 0.383 | 0.528 | -0.145 | 0.0020 | 0.0176 | worse |
| wine | 0.299 | 0.356 | -0.057 | 0.0020 | 0.0176 | worse |
| breast_cancer | 0.290 | 0.412 | -0.121 | 0.0020 | 0.0176 | worse |
| thyroid | 0.190 | 0.213 | -0.023 | 0.1602 | 0.1602 | level |
| shuttle | 0.171 | 0.314 | -0.144 | 0.0020 | 0.0176 | worse |
| letter | 0.424 | 0.366 | +0.058 | 0.0020 | 0.0176 | better |
| landsat | 0.328 | 0.414 | -0.085 | 0.0020 | 0.0176 | worse |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP** (deletion_gain_untied): better 1, level 1, worse 7

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 0.427 | 0.536 | -0.109 | 0.0020 | 0.0176 | worse |
| S2 | 0.394 | 0.556 | -0.162 | 0.0020 | 0.0176 | worse |
| S3 | 0.383 | 0.561 | -0.178 | 0.0020 | 0.0176 | worse |
| wine | 0.299 | 0.528 | -0.229 | 0.0020 | 0.0176 | worse |
| breast_cancer | 0.290 | 0.465 | -0.175 | 0.0020 | 0.0176 | worse |
| thyroid | 0.190 | 0.178 | +0.013 | 0.4316 | 0.4316 | level |
| shuttle | 0.171 | 0.287 | -0.116 | 0.0020 | 0.0176 | worse |
| letter | 0.424 | 0.385 | +0.039 | 0.0020 | 0.0176 | better |
| landsat | 0.328 | 0.532 | -0.204 | 0.0020 | 0.0176 | worse |

Runs whose random-ranking deletion AUC on untied points is within 0.05 of zero: CFMNN + PACE-SHAP 4 of 110, EBM + terms 6 of 110, Tree + TreeSHAP 3 of 110

Untied explained points per run (mean over seeds):

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Tree + TreeSHAP |
|---|---|---|---|
| S1 | 50.0 | 50.0 | 50.0 |
| S2 | 50.0 | 50.0 | 50.0 |
| S3 | 50.0 | 50.0 | 50.0 |
| wine | 36.0 | 36.0 | 36.0 |
| breast_cancer | 49.9 | 50.0 | 50.0 |
| thyroid | 46.9 | 50.0 | 50.0 |
| shuttle | 45.3 | 50.0 | 50.0 |
| magic | 11.5 | 50.0 | 49.9 |
| letter | 23.2 | 50.0 | 48.7 |
| landsat | 37.1 | 50.0 | 49.8 |
| page_blocks | 12.7 | 50.0 | 50.0 |

## Secondary: normalised insertion lead

**CFMNN + PACE-SHAP vs EBM + terms**, per dataset (insertion_lead): better 5, level 5, worse 1

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 1.368 | 1.080 | +0.288 | 0.0020 | 0.0215 | better |
| S2 | 1.488 | 1.077 | +0.412 | 0.0020 | 0.0215 | better |
| S3 | 1.091 | 0.973 | +0.118 | 0.0039 | 0.0273 | better |
| wine | 1.233 | 1.331 | -0.097 | 0.1602 | 0.6406 | level |
| breast_cancer | 0.676 | 0.783 | -0.107 | 0.0039 | 0.0273 | worse |
| thyroid | 0.245 | 0.224 | +0.021 | 0.6953 | 0.9844 | level |
| shuttle | 0.369 | 0.405 | -0.035 | 0.1934 | 0.6406 | level |
| magic | 1.907 | 1.000 | +0.907 | 0.0020 | 0.0215 | better |
| letter | 1.054 | 0.949 | +0.106 | 0.0840 | 0.4199 | level |
| landsat | 10.938 | 21.465 | -10.527 | 0.4922 | 0.9844 | level |
| page_blocks | 2.711 | 0.160 | +2.550 | 0.0020 | 0.0215 | better |

**CFMNN + PACE-SHAP vs EBM + terms**, synthetic family as one unit (insertion_lead): better 3, level 5, worse 1

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 1.316 | 1.043 | +0.273 | 0.0020 | 0.0176 | better |
| wine | 1.233 | 1.331 | -0.097 | 0.1602 | 0.6406 | level |
| breast_cancer | 0.676 | 0.783 | -0.107 | 0.0039 | 0.0234 | worse |
| thyroid | 0.245 | 0.224 | +0.021 | 0.6953 | 0.9844 | level |
| shuttle | 0.369 | 0.405 | -0.035 | 0.1934 | 0.6406 | level |
| magic | 1.907 | 1.000 | +0.907 | 0.0020 | 0.0176 | better |
| letter | 1.054 | 0.949 | +0.106 | 0.0840 | 0.4199 | level |
| landsat | 10.938 | 21.465 | -10.527 | 0.4922 | 0.9844 | level |
| page_blocks | 2.711 | 0.160 | +2.550 | 0.0020 | 0.0176 | better |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, per dataset (insertion_lead): better 6, level 1, worse 4

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 1.368 | 1.105 | +0.263 | 0.0039 | 0.0215 | better |
| S2 | 1.488 | 1.012 | +0.477 | 0.0020 | 0.0215 | better |
| S3 | 1.091 | 0.895 | +0.196 | 0.0020 | 0.0215 | better |
| wine | 1.233 | 2.115 | -0.882 | 0.0020 | 0.0215 | worse |
| breast_cancer | 0.676 | 0.788 | -0.112 | 0.0098 | 0.0215 | worse |
| thyroid | 0.245 | 0.164 | +0.081 | 0.0020 | 0.0215 | better |
| shuttle | 0.369 | 0.400 | -0.031 | 0.1309 | 0.1309 | level |
| magic | 1.907 | 0.992 | +0.915 | 0.0020 | 0.0215 | better |
| letter | 1.054 | 2.097 | -1.043 | 0.0020 | 0.0215 | worse |
| landsat | 10.938 | 23.540 | -12.603 | 0.0020 | 0.0215 | worse |
| page_blocks | 2.711 | 0.199 | +2.512 | 0.0020 | 0.0215 | better |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, synthetic family as one unit (insertion_lead): better 4, level 1, worse 4

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 1.316 | 1.004 | +0.312 | 0.0020 | 0.0176 | better |
| wine | 1.233 | 2.115 | -0.882 | 0.0020 | 0.0176 | worse |
| breast_cancer | 0.676 | 0.788 | -0.112 | 0.0098 | 0.0195 | worse |
| thyroid | 0.245 | 0.164 | +0.081 | 0.0020 | 0.0176 | better |
| shuttle | 0.369 | 0.400 | -0.031 | 0.1309 | 0.1309 | level |
| magic | 1.907 | 0.992 | +0.915 | 0.0020 | 0.0176 | better |
| letter | 1.054 | 2.097 | -1.043 | 0.0020 | 0.0176 | worse |
| landsat | 10.938 | 23.540 | -12.603 | 0.0020 | 0.0176 | worse |
| page_blocks | 2.711 | 0.199 | +2.512 | 0.0020 | 0.0176 | better |

## Secondary: test accuracy (%)

**CFMNN + PACE-SHAP vs EBM + terms**, per dataset (accuracy): better 0, level 1, worse 10

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 89.39 | 99.96 | -10.57 | 0.0020 | 0.0215 | worse |
| S2 | 80.14 | 90.63 | -10.49 | 0.0020 | 0.0215 | worse |
| S3 | 88.06 | 99.91 | -11.85 | 0.0020 | 0.0215 | worse |
| wine | 97.78 | 97.50 | +0.28 | 1.0000 | 1.0000 | level |
| breast_cancer | 95.88 | 97.46 | -1.58 | 0.0039 | 0.0215 | worse |
| thyroid | 96.90 | 99.60 | -2.70 | 0.0020 | 0.0215 | worse |
| shuttle | 99.78 | 99.96 | -0.18 | 0.0020 | 0.0215 | worse |
| magic | 71.88 | 85.97 | -14.09 | 0.0020 | 0.0215 | worse |
| letter | 75.61 | 87.79 | -12.18 | 0.0020 | 0.0215 | worse |
| landsat | 85.41 | 90.64 | -5.23 | 0.0020 | 0.0215 | worse |
| page_blocks | 83.36 | 97.46 | -14.10 | 0.0020 | 0.0215 | worse |

**CFMNN + PACE-SHAP vs EBM + terms**, synthetic family as one unit (accuracy): better 0, level 1, worse 8

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 85.86 | 96.83 | -10.97 | 0.0020 | 0.0176 | worse |
| wine | 97.78 | 97.50 | +0.28 | 1.0000 | 1.0000 | level |
| breast_cancer | 95.88 | 97.46 | -1.58 | 0.0039 | 0.0176 | worse |
| thyroid | 96.90 | 99.60 | -2.70 | 0.0020 | 0.0176 | worse |
| shuttle | 99.78 | 99.96 | -0.18 | 0.0020 | 0.0176 | worse |
| magic | 71.88 | 85.97 | -14.09 | 0.0020 | 0.0176 | worse |
| letter | 75.61 | 87.79 | -12.18 | 0.0020 | 0.0176 | worse |
| landsat | 85.41 | 90.64 | -5.23 | 0.0020 | 0.0176 | worse |
| page_blocks | 83.36 | 97.46 | -14.10 | 0.0020 | 0.0176 | worse |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, per dataset (accuracy): better 2, level 0, worse 9

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 89.39 | 97.14 | -7.75 | 0.0020 | 0.0215 | worse |
| S2 | 80.14 | 88.37 | -8.23 | 0.0020 | 0.0215 | worse |
| S3 | 88.06 | 99.96 | -11.90 | 0.0020 | 0.0215 | worse |
| wine | 97.78 | 92.50 | +5.28 | 0.0078 | 0.0215 | better |
| breast_cancer | 95.88 | 92.37 | +3.51 | 0.0020 | 0.0215 | better |
| thyroid | 96.90 | 99.57 | -2.67 | 0.0020 | 0.0215 | worse |
| shuttle | 99.78 | 99.98 | -0.20 | 0.0020 | 0.0215 | worse |
| magic | 71.88 | 85.10 | -13.22 | 0.0020 | 0.0215 | worse |
| letter | 75.61 | 86.66 | -11.05 | 0.0020 | 0.0215 | worse |
| landsat | 85.41 | 86.59 | -1.18 | 0.0098 | 0.0215 | worse |
| page_blocks | 83.36 | 96.89 | -13.53 | 0.0020 | 0.0215 | worse |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, synthetic family as one unit (accuracy): better 2, level 0, worse 7

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 85.86 | 95.16 | -9.29 | 0.0020 | 0.0176 | worse |
| wine | 97.78 | 92.50 | +5.28 | 0.0078 | 0.0176 | better |
| breast_cancer | 95.88 | 92.37 | +3.51 | 0.0020 | 0.0176 | better |
| thyroid | 96.90 | 99.57 | -2.67 | 0.0020 | 0.0176 | worse |
| shuttle | 99.78 | 99.98 | -0.20 | 0.0020 | 0.0176 | worse |
| magic | 71.88 | 85.10 | -13.22 | 0.0020 | 0.0176 | worse |
| letter | 75.61 | 86.66 | -11.05 | 0.0020 | 0.0176 | worse |
| landsat | 85.41 | 86.59 | -1.18 | 0.0098 | 0.0176 | worse |
| page_blocks | 83.36 | 96.89 | -13.53 | 0.0020 | 0.0176 | worse |

## Secondary: macro-F1 (%)

**CFMNN + PACE-SHAP vs EBM + terms**, per dataset (macro_f1): better 0, level 2, worse 9

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 89.39 | 99.96 | -10.57 | 0.0020 | 0.0215 | worse |
| S2 | 80.13 | 90.63 | -10.49 | 0.0020 | 0.0215 | worse |
| S3 | 86.82 | 99.90 | -13.08 | 0.0020 | 0.0215 | worse |
| wine | 97.85 | 97.54 | +0.31 | 0.5000 | 0.5000 | level |
| breast_cancer | 95.55 | 97.26 | -1.71 | 0.0039 | 0.0215 | worse |
| thyroid | 84.45 | 97.91 | -13.46 | 0.0020 | 0.0215 | worse |
| shuttle | 86.03 | 89.41 | -3.37 | 0.2324 | 0.4648 | level |
| magic | 70.63 | 84.10 | -13.47 | 0.0020 | 0.0215 | worse |
| letter | 75.37 | 87.74 | -12.37 | 0.0020 | 0.0215 | worse |
| landsat | 82.72 | 88.75 | -6.03 | 0.0020 | 0.0215 | worse |
| page_blocks | 59.95 | 86.42 | -26.47 | 0.0020 | 0.0215 | worse |

**CFMNN + PACE-SHAP vs EBM + terms**, synthetic family as one unit (macro_f1): better 0, level 2, worse 7

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 85.45 | 96.83 | -11.38 | 0.0020 | 0.0176 | worse |
| wine | 97.85 | 97.54 | +0.31 | 0.5000 | 0.5000 | level |
| breast_cancer | 95.55 | 97.26 | -1.71 | 0.0039 | 0.0176 | worse |
| thyroid | 84.45 | 97.91 | -13.46 | 0.0020 | 0.0176 | worse |
| shuttle | 86.03 | 89.41 | -3.37 | 0.2324 | 0.4648 | level |
| magic | 70.63 | 84.10 | -13.47 | 0.0020 | 0.0176 | worse |
| letter | 75.37 | 87.74 | -12.37 | 0.0020 | 0.0176 | worse |
| landsat | 82.72 | 88.75 | -6.03 | 0.0020 | 0.0176 | worse |
| page_blocks | 59.95 | 86.42 | -26.47 | 0.0020 | 0.0176 | worse |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, per dataset (macro_f1): better 2, level 1, worse 8

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 89.39 | 97.14 | -7.75 | 0.0020 | 0.0215 | worse |
| S2 | 80.13 | 88.37 | -8.23 | 0.0020 | 0.0215 | worse |
| S3 | 86.82 | 99.96 | -13.13 | 0.0020 | 0.0215 | worse |
| wine | 97.85 | 92.58 | +5.27 | 0.0039 | 0.0215 | better |
| breast_cancer | 95.55 | 91.77 | +3.78 | 0.0020 | 0.0215 | better |
| thyroid | 84.45 | 97.72 | -13.28 | 0.0020 | 0.0215 | worse |
| shuttle | 86.03 | 98.86 | -12.82 | 0.0020 | 0.0215 | worse |
| magic | 70.63 | 83.03 | -12.40 | 0.0020 | 0.0215 | worse |
| letter | 75.37 | 86.66 | -11.30 | 0.0020 | 0.0215 | worse |
| landsat | 82.72 | 83.66 | -0.94 | 0.1055 | 0.1055 | level |
| page_blocks | 59.95 | 81.07 | -21.12 | 0.0020 | 0.0215 | worse |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, synthetic family as one unit (macro_f1): better 2, level 1, worse 6

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 85.45 | 95.15 | -9.71 | 0.0020 | 0.0176 | worse |
| wine | 97.85 | 92.58 | +5.27 | 0.0039 | 0.0176 | better |
| breast_cancer | 95.55 | 91.77 | +3.78 | 0.0020 | 0.0176 | better |
| thyroid | 84.45 | 97.72 | -13.28 | 0.0020 | 0.0176 | worse |
| shuttle | 86.03 | 98.86 | -12.82 | 0.0020 | 0.0176 | worse |
| magic | 70.63 | 83.03 | -12.40 | 0.0020 | 0.0176 | worse |
| letter | 75.37 | 86.66 | -11.30 | 0.0020 | 0.0176 | worse |
| landsat | 82.72 | 83.66 | -0.94 | 0.1055 | 0.1055 | level |
| page_blocks | 59.95 | 81.07 | -21.12 | 0.0020 | 0.0176 | worse |

## Secondary: milliseconds per explanation (lower is better)

**CFMNN + PACE-SHAP vs EBM + terms**, per dataset (ms_per_explanation): better 0, level 0, worse 11

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 5247.718 | 0.021 | +5247.696 | 0.0020 | 0.0215 | worse |
| S2 | 6112.467 | 0.028 | +6112.439 | 0.0020 | 0.0215 | worse |
| S3 | 6015.780 | 0.025 | +6015.755 | 0.0020 | 0.0215 | worse |
| wine | 20.975 | 0.027 | +20.948 | 0.0020 | 0.0215 | worse |
| breast_cancer | 172.091 | 0.047 | +172.044 | 0.0020 | 0.0215 | worse |
| thyroid | 118.177 | 0.026 | +118.151 | 0.0020 | 0.0215 | worse |
| shuttle | 21.359 | 0.025 | +21.334 | 0.0020 | 0.0215 | worse |
| magic | 761.792 | 0.025 | +761.767 | 0.0020 | 0.0215 | worse |
| letter | 2398.209 | 0.043 | +2398.166 | 0.0020 | 0.0215 | worse |
| landsat | 2387.541 | 0.066 | +2387.474 | 0.0020 | 0.0215 | worse |
| page_blocks | 48.161 | 0.013 | +48.147 | 0.0020 | 0.0215 | worse |

**CFMNN + PACE-SHAP vs EBM + terms**, synthetic family as one unit (ms_per_explanation): better 0, level 0, worse 9

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 5791.988 | 0.025 | +5791.963 | 0.0020 | 0.0176 | worse |
| wine | 20.975 | 0.027 | +20.948 | 0.0020 | 0.0176 | worse |
| breast_cancer | 172.091 | 0.047 | +172.044 | 0.0020 | 0.0176 | worse |
| thyroid | 118.177 | 0.026 | +118.151 | 0.0020 | 0.0176 | worse |
| shuttle | 21.359 | 0.025 | +21.334 | 0.0020 | 0.0176 | worse |
| magic | 761.792 | 0.025 | +761.767 | 0.0020 | 0.0176 | worse |
| letter | 2398.209 | 0.043 | +2398.166 | 0.0020 | 0.0176 | worse |
| landsat | 2387.541 | 0.066 | +2387.474 | 0.0020 | 0.0176 | worse |
| page_blocks | 48.161 | 0.013 | +48.147 | 0.0020 | 0.0176 | worse |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, per dataset (ms_per_explanation): better 0, level 0, worse 11

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 5247.718 | 0.148 | +5247.569 | 0.0020 | 0.0215 | worse |
| S2 | 6112.467 | 0.120 | +6112.347 | 0.0020 | 0.0215 | worse |
| S3 | 6015.780 | 0.056 | +6015.724 | 0.0020 | 0.0215 | worse |
| wine | 20.975 | 0.057 | +20.918 | 0.0020 | 0.0215 | worse |
| breast_cancer | 172.091 | 0.072 | +172.019 | 0.0020 | 0.0215 | worse |
| thyroid | 118.177 | 0.064 | +118.113 | 0.0020 | 0.0215 | worse |
| shuttle | 21.359 | 6.301 | +15.058 | 0.0039 | 0.0215 | worse |
| magic | 761.792 | 0.244 | +761.548 | 0.0020 | 0.0215 | worse |
| letter | 2398.209 | 41.768 | +2356.441 | 0.0020 | 0.0215 | worse |
| landsat | 2387.541 | 0.830 | +2386.711 | 0.0020 | 0.0215 | worse |
| page_blocks | 48.161 | 0.123 | +48.038 | 0.0020 | 0.0215 | worse |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, synthetic family as one unit (ms_per_explanation): better 0, level 0, worse 9

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 5791.988 | 0.108 | +5791.880 | 0.0020 | 0.0176 | worse |
| wine | 20.975 | 0.057 | +20.918 | 0.0020 | 0.0176 | worse |
| breast_cancer | 172.091 | 0.072 | +172.019 | 0.0020 | 0.0176 | worse |
| thyroid | 118.177 | 0.064 | +118.113 | 0.0020 | 0.0176 | worse |
| shuttle | 21.359 | 6.301 | +15.058 | 0.0039 | 0.0176 | worse |
| magic | 761.792 | 0.244 | +761.548 | 0.0020 | 0.0176 | worse |
| letter | 2398.209 | 41.768 | +2356.441 | 0.0020 | 0.0176 | worse |
| landsat | 2387.541 | 0.830 | +2386.711 | 0.0020 | 0.0176 | worse |
| page_blocks | 48.161 | 0.123 | +48.038 | 0.0020 | 0.0176 | worse |

## Sensitivity: deletion gain (random minus method, not divided)

**CFMNN + PACE-SHAP vs EBM + terms**, per dataset (deletion_gain): better 0, level 1, worse 10

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 0.427 | 0.524 | -0.097 | 0.0020 | 0.0215 | worse |
| S2 | 0.394 | 0.536 | -0.142 | 0.0020 | 0.0215 | worse |
| S3 | 0.383 | 0.528 | -0.145 | 0.0020 | 0.0215 | worse |
| wine | 0.299 | 0.356 | -0.057 | 0.0020 | 0.0215 | worse |
| breast_cancer | 0.290 | 0.412 | -0.122 | 0.0020 | 0.0215 | worse |
| thyroid | 0.203 | 0.213 | -0.010 | 0.5566 | 0.5566 | level |
| shuttle | 0.172 | 0.314 | -0.143 | 0.0020 | 0.0215 | worse |
| magic | 0.171 | 0.595 | -0.424 | 0.0020 | 0.0215 | worse |
| letter | 0.302 | 0.366 | -0.064 | 0.0020 | 0.0215 | worse |
| landsat | 0.296 | 0.414 | -0.117 | 0.0020 | 0.0215 | worse |
| page_blocks | 0.124 | 0.172 | -0.048 | 0.0098 | 0.0215 | worse |

**CFMNN + PACE-SHAP vs EBM + terms**, synthetic family as one unit (deletion_gain): better 0, level 1, worse 8

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 0.401 | 0.529 | -0.128 | 0.0020 | 0.0176 | worse |
| wine | 0.299 | 0.356 | -0.057 | 0.0020 | 0.0176 | worse |
| breast_cancer | 0.290 | 0.412 | -0.122 | 0.0020 | 0.0176 | worse |
| thyroid | 0.203 | 0.213 | -0.010 | 0.5566 | 0.5566 | level |
| shuttle | 0.172 | 0.314 | -0.143 | 0.0020 | 0.0176 | worse |
| magic | 0.171 | 0.595 | -0.424 | 0.0020 | 0.0176 | worse |
| letter | 0.302 | 0.366 | -0.064 | 0.0020 | 0.0176 | worse |
| landsat | 0.296 | 0.414 | -0.117 | 0.0020 | 0.0176 | worse |
| page_blocks | 0.124 | 0.172 | -0.048 | 0.0098 | 0.0195 | worse |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, per dataset (deletion_gain): better 0, level 2, worse 9

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 0.427 | 0.536 | -0.109 | 0.0020 | 0.0215 | worse |
| S2 | 0.394 | 0.556 | -0.162 | 0.0020 | 0.0215 | worse |
| S3 | 0.383 | 0.561 | -0.178 | 0.0020 | 0.0215 | worse |
| wine | 0.299 | 0.528 | -0.229 | 0.0020 | 0.0215 | worse |
| breast_cancer | 0.290 | 0.465 | -0.175 | 0.0020 | 0.0215 | worse |
| thyroid | 0.203 | 0.178 | +0.026 | 0.1309 | 0.2617 | level |
| shuttle | 0.172 | 0.287 | -0.115 | 0.0020 | 0.0215 | worse |
| magic | 0.171 | 0.552 | -0.382 | 0.0020 | 0.0215 | worse |
| letter | 0.302 | 0.378 | -0.076 | 0.0020 | 0.0215 | worse |
| landsat | 0.296 | 0.532 | -0.235 | 0.0020 | 0.0215 | worse |
| page_blocks | 0.124 | 0.154 | -0.030 | 0.2324 | 0.2617 | level |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, synthetic family as one unit (deletion_gain): better 0, level 2, worse 7

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 0.401 | 0.551 | -0.149 | 0.0020 | 0.0176 | worse |
| wine | 0.299 | 0.528 | -0.229 | 0.0020 | 0.0176 | worse |
| breast_cancer | 0.290 | 0.465 | -0.175 | 0.0020 | 0.0176 | worse |
| thyroid | 0.203 | 0.178 | +0.026 | 0.1309 | 0.2617 | level |
| shuttle | 0.172 | 0.287 | -0.115 | 0.0020 | 0.0176 | worse |
| magic | 0.171 | 0.552 | -0.382 | 0.0020 | 0.0176 | worse |
| letter | 0.302 | 0.378 | -0.076 | 0.0020 | 0.0176 | worse |
| landsat | 0.296 | 0.532 | -0.235 | 0.0020 | 0.0176 | worse |
| page_blocks | 0.124 | 0.154 | -0.030 | 0.2324 | 0.2617 | level |

## Sensitivity: insertion gain (method minus random, not divided)

**CFMNN + PACE-SHAP vs EBM + terms**, per dataset (insertion_gain): better 1, level 2, worse 8

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 0.373 | 0.408 | -0.035 | 0.0020 | 0.0215 | worse |
| S2 | 0.362 | 0.406 | -0.044 | 0.0020 | 0.0215 | worse |
| S3 | 0.329 | 0.442 | -0.112 | 0.0020 | 0.0215 | worse |
| wine | 0.293 | 0.353 | -0.060 | 0.0020 | 0.0215 | worse |
| breast_cancer | 0.281 | 0.351 | -0.070 | 0.0020 | 0.0215 | worse |
| thyroid | 0.176 | 0.175 | +0.001 | 1.0000 | 1.0000 | level |
| shuttle | 0.203 | 0.245 | -0.042 | 0.0059 | 0.0215 | worse |
| magic | 0.175 | 0.408 | -0.233 | 0.0020 | 0.0215 | worse |
| letter | 0.418 | 0.339 | +0.080 | 0.0020 | 0.0215 | better |
| landsat | 0.279 | 0.351 | -0.073 | 0.0020 | 0.0215 | worse |
| page_blocks | 0.146 | 0.114 | +0.033 | 0.0488 | 0.0977 | level |

**CFMNN + PACE-SHAP vs EBM + terms**, synthetic family as one unit (insertion_gain): better 1, level 2, worse 6

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 0.355 | 0.419 | -0.064 | 0.0020 | 0.0176 | worse |
| wine | 0.293 | 0.353 | -0.060 | 0.0020 | 0.0176 | worse |
| breast_cancer | 0.281 | 0.351 | -0.070 | 0.0020 | 0.0176 | worse |
| thyroid | 0.176 | 0.175 | +0.001 | 1.0000 | 1.0000 | level |
| shuttle | 0.203 | 0.245 | -0.042 | 0.0059 | 0.0176 | worse |
| magic | 0.175 | 0.408 | -0.233 | 0.0020 | 0.0176 | worse |
| letter | 0.418 | 0.339 | +0.080 | 0.0020 | 0.0176 | better |
| landsat | 0.279 | 0.351 | -0.073 | 0.0020 | 0.0176 | worse |
| page_blocks | 0.146 | 0.114 | +0.033 | 0.0488 | 0.0977 | level |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, per dataset (insertion_gain): better 0, level 2, worse 9

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| S1 | 0.373 | 0.456 | -0.083 | 0.0020 | 0.0215 | worse |
| S2 | 0.362 | 0.420 | -0.058 | 0.0020 | 0.0215 | worse |
| S3 | 0.329 | 0.431 | -0.101 | 0.0020 | 0.0215 | worse |
| wine | 0.293 | 0.578 | -0.285 | 0.0020 | 0.0215 | worse |
| breast_cancer | 0.281 | 0.423 | -0.142 | 0.0020 | 0.0215 | worse |
| thyroid | 0.176 | 0.136 | +0.040 | 0.0371 | 0.0742 | level |
| shuttle | 0.203 | 0.259 | -0.056 | 0.0039 | 0.0215 | worse |
| magic | 0.175 | 0.398 | -0.222 | 0.0020 | 0.0215 | worse |
| letter | 0.418 | 0.769 | -0.351 | 0.0020 | 0.0215 | worse |
| landsat | 0.279 | 0.773 | -0.495 | 0.0020 | 0.0215 | worse |
| page_blocks | 0.146 | 0.153 | -0.007 | 0.6953 | 0.6953 | level |

**CFMNN + PACE-SHAP vs Tree + TreeSHAP**, synthetic family as one unit (insertion_gain): better 0, level 2, worse 7

| Dataset | CFMNN + PACE-SHAP | Tree + TreeSHAP | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Synthetic family | 0.355 | 0.436 | -0.081 | 0.0020 | 0.0176 | worse |
| wine | 0.293 | 0.578 | -0.285 | 0.0020 | 0.0176 | worse |
| breast_cancer | 0.281 | 0.423 | -0.142 | 0.0020 | 0.0176 | worse |
| thyroid | 0.176 | 0.136 | +0.040 | 0.0371 | 0.0742 | level |
| shuttle | 0.203 | 0.259 | -0.056 | 0.0039 | 0.0176 | worse |
| magic | 0.175 | 0.398 | -0.222 | 0.0020 | 0.0176 | worse |
| letter | 0.418 | 0.769 | -0.351 | 0.0020 | 0.0176 | worse |
| landsat | 0.279 | 0.773 | -0.495 | 0.0020 | 0.0176 | worse |
| page_blocks | 0.146 | 0.153 | -0.007 | 0.6953 | 0.6953 | level |

## Descriptive

## Size: rules (boxes, terms, leaves)

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Tree + TreeSHAP |
|---|---|---|---|
| S1 | 1516.0 | 10.0 | 57.7 |
| S2 | 1537.7 | 10.0 | 32.0 |
| S3 | 1528.8 | 10.0 | 4.0 |
| wine | 31.1 | 13.0 | 6.5 |
| breast_cancer | 41.8 | 30.0 | 6.7 |
| thyroid | 53.3 | 21.0 | 16.2 |
| shuttle | 14.3 | 9.0 | 31.8 |
| magic | 284.4 | 10.0 | 96.0 |
| letter | 563.3 | 16.0 | 1024.0 |
| landsat | 199.8 | 36.0 | 83.2 |
| page_blocks | 26.2 | 10.0 | 22.4 |

## Conditions per rule

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Tree + TreeSHAP |
|---|---|---|---|
| S1 | 10.00 | 1.00 | 3.32 |
| S2 | 10.00 | 1.00 | 3.04 |
| S3 | 10.00 | 1.00 | 2.25 |
| wine | 13.00 | 1.00 | 2.80 |
| breast_cancer | 30.00 | 1.00 | 2.51 |
| thyroid | 16.61 | 1.00 | 4.69 |
| shuttle | 9.00 | 1.00 | 3.82 |
| magic | 10.00 | 1.00 | 5.11 |
| letter | 16.00 | 1.00 | 9.30 |
| landsat | 36.00 | 1.00 | 6.25 |
| page_blocks | 9.97 | 1.00 | 3.98 |

## Deletion margin AUC of the system (lower is better)

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Tree + TreeSHAP |
|---|---|---|---|
| S1 | -0.142 | -0.127 | -0.102 |
| S2 | -0.142 | -0.146 | -0.122 |
| S3 | -0.071 | -0.069 | -0.077 |
| wine | -0.048 | -0.068 | -0.240 |
| breast_cancer | 0.128 | 0.038 | 0.075 |
| thyroid | 0.532 | 0.594 | 0.680 |
| shuttle | 0.405 | 0.298 | 0.378 |
| magic | -0.041 | -0.183 | -0.138 |
| letter | -0.687 | -0.724 | -0.744 |
| landsat | -0.312 | -0.362 | -0.475 |
| page_blocks | 0.016 | 0.548 | 0.638 |

## Deletion margin AUC of a random ranking on the same model

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Tree + TreeSHAP |
|---|---|---|---|
| S1 | 0.284 | 0.397 | 0.434 |
| S2 | 0.252 | 0.390 | 0.433 |
| S3 | 0.312 | 0.459 | 0.483 |
| wine | 0.250 | 0.288 | 0.288 |
| breast_cancer | 0.418 | 0.449 | 0.540 |
| thyroid | 0.735 | 0.807 | 0.857 |
| shuttle | 0.577 | 0.613 | 0.665 |
| magic | 0.130 | 0.412 | 0.415 |
| letter | -0.385 | -0.358 | -0.366 |
| landsat | -0.016 | 0.051 | 0.057 |
| page_blocks | 0.140 | 0.720 | 0.793 |

## Insertion margin AUC of the system (higher is better)

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Tree + TreeSHAP |
|---|---|---|---|
| S1 | 0.650 | 0.791 | 0.875 |
| S2 | 0.613 | 0.785 | 0.838 |
| S3 | 0.639 | 0.900 | 0.916 |
| wine | 0.532 | 0.624 | 0.861 |
| breast_cancer | 0.698 | 0.801 | 0.960 |
| thyroid | 0.905 | 0.972 | 0.988 |
| shuttle | 0.770 | 0.857 | 0.917 |
| magic | 0.275 | 0.821 | 0.804 |
| letter | 0.020 | -0.020 | 0.402 |
| landsat | 0.251 | 0.388 | 0.805 |
| page_blocks | 0.244 | 0.833 | 0.941 |

## Insertion margin AUC of a random ranking on the same model

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Tree + TreeSHAP |
|---|---|---|---|
| S1 | 0.276 | 0.383 | 0.418 |
| S2 | 0.251 | 0.380 | 0.418 |
| S3 | 0.310 | 0.458 | 0.486 |
| wine | 0.240 | 0.271 | 0.284 |
| breast_cancer | 0.418 | 0.450 | 0.537 |
| thyroid | 0.729 | 0.798 | 0.852 |
| shuttle | 0.567 | 0.612 | 0.657 |
| magic | 0.099 | 0.412 | 0.406 |
| letter | -0.398 | -0.358 | -0.367 |
| landsat | -0.027 | 0.036 | 0.032 |
| page_blocks | 0.098 | 0.719 | 0.787 |

## Share of explained points whose margin is zero

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Tree + TreeSHAP |
|---|---|---|---|
| S1 | 0.000 | 0.000 | 0.000 |
| S2 | 0.000 | 0.000 | 0.000 |
| S3 | 0.000 | 0.000 | 0.000 |
| wine | 0.000 | 0.000 | 0.000 |
| breast_cancer | 0.002 | 0.000 | 0.000 |
| thyroid | 0.062 | 0.000 | 0.000 |
| shuttle | 0.094 | 0.000 | 0.000 |
| magic | 0.770 | 0.000 | 0.002 |
| letter | 0.536 | 0.000 | 0.026 |
| landsat | 0.258 | 0.000 | 0.004 |
| page_blocks | 0.746 | 0.000 | 0.000 |

Runs whose random-ranking deletion AUC is within 0.05 of zero (lead unstable): CFMNN + PACE-SHAP 8 of 110, EBM + terms 6 of 110, Tree + TreeSHAP 4 of 110

Leaves the tree was allowed (count over seeds):

- shuttle: 32 ×6, 64 ×4
- letter: 1024 ×10
- S1: 64 ×5, 32 ×3, 128 ×2
- S2: 32 ×10
- S3: 4 ×8, 8 ×2
- magic: 128 ×5, 64 ×5
- wine: 8 ×7, 4 ×2, 16 ×1
- breast_cancer: 4 ×5, 8 ×4, 16 ×1
- landsat: 64 ×7, 128 ×3
- thyroid: 16 ×5, 32 ×3, 8 ×2
- page_blocks: 16 ×6, 32 ×4

## Amendment 1: milliseconds per explanation, measured alone (seeds [120])

One task at a time with nothing else running. These are the times to quote; the times in the run above were taken while 12 tasks shared the CPU.

| Dataset | CFMNN + PACE-SHAP | EBM + terms | Tree + TreeSHAP | CFMNN + PACE-SHAP ÷ EBM + terms | CFMNN + PACE-SHAP ÷ Tree + TreeSHAP |
|---|---|---|---|---|---|
| S1 | 1,774.757 | 0.011 | 0.066 | 159,916.8× | 26,815.5× |
| S2 | 1,812.244 | 0.011 | 0.071 | 166,750.4× | 25,410.0× |
| S3 | 1,801.704 | 0.011 | 0.026 | 169,110.6× | 68,531.9× |
| wine | 6.693 | 0.013 | 0.026 | 525.2× | 255.6× |
| breast_cancer | 85.365 | 0.022 | 0.029 | 3,874.2× | 2,961.4× |
| thyroid | 59.030 | 0.015 | 0.035 | 3,884.1× | 1,708.3× |
| shuttle | 6.593 | 0.011 | 0.188 | 605.8× | 35.1× |
| magic | 248.362 | 0.011 | 0.153 | 23,294.2× | 1,625.7× |
| letter | 507.509 | 0.019 | 8.458 | 27,279.6× | 60.0× |
| landsat | 635.736 | 0.027 | 0.430 | 23,467.6× | 1,478.9× |
| page_blocks | 22.452 | 0.010 | 0.080 | 2,140.7× | 281.7× |

