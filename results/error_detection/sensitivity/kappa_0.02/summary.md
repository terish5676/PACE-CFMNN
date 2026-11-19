# RQ2: error detection by the trust signals

Main model (fixed centroid + search). Role split 50/20/10/10/10; PRS ranks fitted on the inner-validation rows; errors scored on all other rows. Perturbations: kappa = 0.02, K = 50. Seeds [0, 1, 2, 3, 4]. Mean ± 95% CI over seeds. A dataset is eligible for RQ2 if every seed has at least 50 errors in its evaluation rows.

Methods: PRS = mean percentile rank of PD, EI and U (PD + U where EI is missing); PD + U = the same without EI; Margin = 1 - (s'1 - s'2); Entropy = normalised entropy of s'; EI alone is scored only on points where it exists.

Starred methods use U*, a corrected U added after gate G1 (a variant, not the planned analysis): the blueprint's U ranks points where the top two classes tie below most other points, although ties are where CFMNN errs most. U* ranks every tie first and grades ties by the centroid-distance ratio.

## Datasets

| Dataset | Eligible | Accuracy (%) | Errors (min over seeds) | Evaluation rows | Boxes | Tied points | Error rate in ties | EI coverage | Mode switches | Box switches |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | yes | 79.8 ± 1.6 | 367 | 2000 | 989 | 0.000 | nan | 1.00 | 0.000 | 0.069 |
| thyroid | yes | 96.7 ± 0.4 | 83 | 2880 | 47 | 0.047 | 0.351 | 1.00 | 0.008 | 0.159 |

## AUPRC (higher is better; compare with the base error rate)

| Dataset | Base error rate | PRS | PD + U | PD | EI | U | PRS* | PD + U* | U* | Margin | Entropy | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 0.202 | 0.298 ± 0.022 | 0.342 ± 0.029 | 0.284 ± 0.028 | 0.216 ± 0.020 | 0.324 ± 0.026 | 0.348 ± 0.020 | **0.406 ± 0.034** | 0.403 ± 0.033 | 0.403 ± 0.033 | 0.405 ± 0.034 | 0.210 ± 0.021 |
| thyroid | 0.033 | 0.099 ± 0.031 | 0.104 ± 0.066 | 0.313 ± 0.029 | 0.228 ± 0.041 | 0.062 ± 0.031 | 0.286 ± 0.050 | 0.296 ± 0.072 | **0.342 ± 0.084** | 0.224 ± 0.053 | 0.084 ± 0.036 | 0.036 ± 0.002 |

## AUROC (0.5 = random)

| Dataset | Base error rate | PRS | PD + U | PD | EI | U | PRS* | PD + U* | U* | Margin | Entropy | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 0.202 | 0.639 ± 0.017 | 0.689 ± 0.012 | 0.596 ± 0.010 | 0.513 ± 0.018 | 0.684 ± 0.012 | 0.691 ± 0.015 | 0.753 ± 0.011 | 0.753 ± 0.011 | 0.753 ± 0.011 | **0.756 ± 0.011** | 0.502 ± 0.012 |
| thyroid | 0.033 | 0.705 ± 0.033 | 0.383 ± 0.066 | **0.892 ± 0.018** | 0.877 ± 0.026 | 0.322 ± 0.059 | 0.880 ± 0.027 | 0.758 ± 0.050 | 0.733 ± 0.055 | 0.729 ± 0.056 | 0.622 ± 0.067 | 0.504 ± 0.025 |

## Precision when the top 10% are flagged

| Dataset | Base error rate | PRS | PD + U | PD | EI | U | PRS* | PD + U* | U* | Margin | Entropy | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 0.202 | 0.360 ± 0.017 | 0.392 ± 0.054 | **0.477 ± 0.038** | 0.233 ± 0.029 | 0.363 ± 0.054 | 0.416 ± 0.014 | 0.469 ± 0.047 | 0.468 ± 0.041 | 0.468 ± 0.041 | 0.469 ± 0.051 | 0.214 ± 0.042 |
| thyroid | 0.033 | 0.081 ± 0.023 | 0.060 ± 0.020 | **0.257 ± 0.039** | 0.186 ± 0.034 | 0.056 ± 0.024 | 0.206 ± 0.031 | 0.200 ± 0.032 | 0.193 ± 0.031 | 0.193 ± 0.031 | 0.078 ± 0.024 | 0.034 ± 0.010 |

## Primary endpoint: does EI add value? AUPRC of PRS minus PD + U

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | -0.044 ± 0.014 | 0 of 5 | 0.0625 | 0.125 | -3.91 |
| thyroid | -0.006 ± 0.045 | 3 of 5 | 1 | 1 | -0.16 |

## Variant: does EI add value with U*? AUPRC of PRS* minus PD + U*

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | -0.057 ± 0.018 | 0 of 5 | 0.0625 | 0.125 | -3.92 |
| thyroid | -0.009 ± 0.039 | 3 of 5 | 0.812 | 0.812 | -0.30 |

## PRS* minus the plain margin (AUPRC)

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | -0.055 ± 0.018 | 0 of 5 | 0.0625 | 0.125 | -3.89 |
| thyroid | 0.062 ± 0.029 | 5 of 5 | 0.0625 | 0.125 | 2.63 |

## PD + U* minus the plain margin (AUPRC)

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | 0.002 ± 0.004 | 4 of 5 | 0.312 | 0.312 | 0.66 |
| thyroid | 0.071 ± 0.041 | 5 of 5 | 0.0625 | 0.125 | 2.14 |

## Signal checks (gate G1)

Plausibility = share of perturbations whose L1 distance to the nearest training row is at most the 95th percentile of the original points' distances (near 0.95 or above = perturbations sit where real data sits). Spread = standard deviation over evaluation points.

| Dataset | Clipped coordinates | Plausibility | Spread of U | Spread of entropy | rho(PD, U) | rho(PD, EI) | rho(EI, U) | Signal time (s) |
|---|---|---|---|---|---|---|---|---|
| S2 | 0.005 | 0.95 | 0.039 | 0.002 | 0.42 | 0.01 | 0.05 | 31.7 |
| thyroid | 0.013 | 0.94 | 0.030 | 0.000 | -0.14 | 0.40 | -0.40 | 5.4 |
