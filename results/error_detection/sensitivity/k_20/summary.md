# RQ2: error detection by the trust signals

Main model (fixed centroid + search). Role split 50/20/10/10/10; PRS ranks fitted on the inner-validation rows; errors scored on all other rows. Perturbations: kappa = 0.05, K = 20. Seeds [0, 1, 2, 3, 4]. Mean ± 95% CI over seeds. A dataset is eligible for RQ2 if every seed has at least 50 errors in its evaluation rows.

Methods: PRS = mean percentile rank of PD, EI and U (PD + U where EI is missing); PD + U = the same without EI; Margin = 1 - (s'1 - s'2); Entropy = normalised entropy of s'; EI alone is scored only on points where it exists.

Starred methods use U*, a corrected U added after gate G1 (a variant, not the planned analysis): the blueprint's U ranks points where the top two classes tie below most other points, although ties are where CFMNN errs most. U* ranks every tie first and grades ties by the centroid-distance ratio.

## Datasets

| Dataset | Eligible | Accuracy (%) | Errors (min over seeds) | Evaluation rows | Boxes | Tied points | Error rate in ties | EI coverage | Mode switches | Box switches |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | yes | 79.8 ± 1.6 | 367 | 2000 | 989 | 0.000 | nan | 1.00 | 0.000 | 0.161 |
| thyroid | yes | 96.7 ± 0.4 | 83 | 2880 | 47 | 0.047 | 0.351 | 1.00 | 0.015 | 0.205 |

## AUPRC (higher is better; compare with the base error rate)

| Dataset | Base error rate | PRS | PD + U | PD | EI | U | PRS* | PD + U* | U* | Margin | Entropy | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 0.202 | 0.320 ± 0.024 | 0.368 ± 0.029 | 0.355 ± 0.039 | 0.221 ± 0.018 | 0.324 ± 0.026 | 0.375 ± 0.023 | **0.422 ± 0.038** | 0.403 ± 0.033 | 0.403 ± 0.033 | 0.405 ± 0.034 | 0.210 ± 0.021 |
| thyroid | 0.033 | 0.076 ± 0.022 | 0.089 ± 0.067 | 0.204 ± 0.037 | 0.210 ± 0.048 | 0.062 ± 0.031 | 0.215 ± 0.052 | 0.222 ± 0.063 | **0.342 ± 0.084** | 0.224 ± 0.053 | 0.084 ± 0.036 | 0.036 ± 0.002 |

## AUROC (0.5 = random)

| Dataset | Base error rate | PRS | PD + U | PD | EI | U | PRS* | PD + U* | U* | Margin | Entropy | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 0.202 | 0.661 ± 0.021 | 0.702 ± 0.013 | 0.668 ± 0.014 | 0.526 ± 0.024 | 0.684 ± 0.012 | 0.711 ± 0.019 | **0.757 ± 0.012** | 0.753 ± 0.011 | 0.753 ± 0.011 | 0.756 ± 0.011 | 0.502 ± 0.012 |
| thyroid | 0.033 | 0.663 ± 0.031 | 0.364 ± 0.061 | 0.822 ± 0.020 | **0.869 ± 0.027** | 0.322 ± 0.059 | 0.864 ± 0.027 | 0.749 ± 0.047 | 0.733 ± 0.055 | 0.729 ± 0.056 | 0.622 ± 0.067 | 0.504 ± 0.025 |

## Precision when the top 10% are flagged

| Dataset | Base error rate | PRS | PD + U | PD | EI | U | PRS* | PD + U* | U* | Margin | Entropy | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 0.202 | 0.390 ± 0.028 | 0.454 ± 0.059 | **0.524 ± 0.040** | 0.223 ± 0.020 | 0.363 ± 0.054 | 0.456 ± 0.021 | 0.494 ± 0.034 | 0.468 ± 0.041 | 0.468 ± 0.041 | 0.469 ± 0.051 | 0.214 ± 0.042 |
| thyroid | 0.033 | 0.074 ± 0.020 | 0.061 ± 0.018 | **0.217 ± 0.039** | 0.183 ± 0.029 | 0.056 ± 0.024 | 0.196 ± 0.035 | 0.200 ± 0.033 | 0.193 ± 0.031 | 0.193 ± 0.031 | 0.078 ± 0.024 | 0.034 ± 0.010 |

## Primary endpoint: does EI add value? AUPRC of PRS minus PD + U

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | -0.048 ± 0.013 | 0 of 5 | 0.0625 | 0.125 | -4.56 |
| thyroid | -0.014 ± 0.057 | 3 of 5 | 1 | 1 | -0.30 |

## Variant: does EI add value with U*? AUPRC of PRS* minus PD + U*

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | -0.048 ± 0.024 | 0 of 5 | 0.0625 | 0.125 | -2.47 |
| thyroid | -0.008 ± 0.017 | 2 of 5 | 0.438 | 0.438 | -0.57 |

## PRS* minus the plain margin (AUPRC)

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | -0.029 ± 0.022 | 0 of 5 | 0.0625 | 0.125 | -1.63 |
| thyroid | -0.010 ± 0.017 | 1 of 5 | 0.188 | 0.188 | -0.71 |

## PD + U* minus the plain margin (AUPRC)

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | 0.019 ± 0.013 | 5 of 5 | 0.0625 | 0.125 | 1.83 |
| thyroid | -0.002 ± 0.015 | 3 of 5 | 0.812 | 0.812 | -0.17 |

## Signal checks (gate G1)

Plausibility = share of perturbations whose L1 distance to the nearest training row is at most the 95th percentile of the original points' distances (near 0.95 or above = perturbations sit where real data sits). Spread = standard deviation over evaluation points.

| Dataset | Clipped coordinates | Plausibility | Spread of U | Spread of entropy | rho(PD, U) | rho(PD, EI) | rho(EI, U) | Signal time (s) |
|---|---|---|---|---|---|---|---|---|
| S2 | 0.012 | 0.95 | 0.039 | 0.002 | 0.55 | 0.01 | 0.09 | 13.8 |
| thyroid | 0.025 | 0.94 | 0.030 | 0.000 | -0.02 | 0.45 | -0.33 | 2.6 |
