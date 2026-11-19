# RQ2: error detection by the trust signals

Main model (fixed centroid + search). Role split 50/20/10/10/10; PRS ranks fitted on the inner-validation rows; errors scored on all other rows. Perturbations: kappa = 0.05, K = 100. Seeds [0, 1, 2, 3, 4]. Mean ± 95% CI over seeds. A dataset is eligible for RQ2 if every seed has at least 50 errors in its evaluation rows.

Methods: PRS = mean percentile rank of PD, EI and U (PD + U where EI is missing); PD + U = the same without EI; Margin = 1 - (s'1 - s'2); Entropy = normalised entropy of s'; EI alone is scored only on points where it exists.

Starred methods use U*, a corrected U added after gate G1 (a variant, not the planned analysis): the blueprint's U ranks points where the top two classes tie below most other points, although ties are where CFMNN errs most. U* ranks every tie first and grades ties by the centroid-distance ratio.

## Datasets

| Dataset | Eligible | Accuracy (%) | Errors (min over seeds) | Evaluation rows | Boxes | Tied points | Error rate in ties | EI coverage | Mode switches | Box switches |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | yes | 79.8 ± 1.6 | 367 | 2000 | 989 | 0.000 | nan | 1.00 | 0.000 | 0.160 |
| thyroid | yes | 96.7 ± 0.4 | 83 | 2880 | 47 | 0.047 | 0.351 | 1.00 | 0.015 | 0.206 |

## AUPRC (higher is better; compare with the base error rate)

| Dataset | Base error rate | PRS | PD + U | PD | EI | U | PRS* | PD + U* | U* | Margin | Entropy | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 0.202 | 0.340 ± 0.026 | 0.383 ± 0.024 | 0.388 ± 0.027 | 0.221 ± 0.017 | 0.324 ± 0.026 | 0.387 ± 0.025 | **0.424 ± 0.031** | 0.403 ± 0.033 | 0.403 ± 0.033 | 0.405 ± 0.034 | 0.210 ± 0.021 |
| thyroid | 0.033 | 0.091 ± 0.028 | 0.089 ± 0.058 | 0.255 ± 0.050 | 0.221 ± 0.040 | 0.062 ± 0.031 | 0.226 ± 0.055 | 0.221 ± 0.060 | **0.342 ± 0.084** | 0.224 ± 0.053 | 0.084 ± 0.036 | 0.036 ± 0.002 |

## AUROC (0.5 = random)

| Dataset | Base error rate | PRS | PD + U | PD | EI | U | PRS* | PD + U* | U* | Margin | Entropy | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 0.202 | 0.673 ± 0.023 | 0.709 ± 0.013 | 0.701 ± 0.012 | 0.531 ± 0.027 | 0.684 ± 0.012 | 0.720 ± 0.019 | **0.758 ± 0.012** | 0.753 ± 0.011 | 0.753 ± 0.011 | 0.756 ± 0.011 | 0.502 ± 0.012 |
| thyroid | 0.033 | 0.762 ± 0.028 | 0.422 ± 0.065 | **0.910 ± 0.019** | 0.892 ± 0.027 | 0.322 ± 0.059 | 0.892 ± 0.026 | 0.773 ± 0.046 | 0.733 ± 0.055 | 0.729 ± 0.056 | 0.622 ± 0.067 | 0.504 ± 0.025 |

## Precision when the top 10% are flagged

| Dataset | Base error rate | PRS | PD + U | PD | EI | U | PRS* | PD + U* | U* | Margin | Entropy | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 0.202 | 0.412 ± 0.027 | 0.467 ± 0.036 | **0.518 ± 0.020** | 0.227 ± 0.014 | 0.363 ± 0.054 | 0.472 ± 0.026 | 0.484 ± 0.030 | 0.468 ± 0.041 | 0.468 ± 0.041 | 0.469 ± 0.051 | 0.214 ± 0.042 |
| thyroid | 0.033 | 0.080 ± 0.017 | 0.062 ± 0.017 | **0.259 ± 0.034** | 0.188 ± 0.027 | 0.056 ± 0.024 | 0.206 ± 0.032 | 0.202 ± 0.034 | 0.193 ± 0.031 | 0.193 ± 0.031 | 0.078 ± 0.024 | 0.034 ± 0.010 |

## Primary endpoint: does EI add value? AUPRC of PRS minus PD + U

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | -0.044 ± 0.022 | 0 of 5 | 0.0625 | 0.125 | -2.52 |
| thyroid | 0.002 ± 0.041 | 4 of 5 | 0.625 | 0.625 | 0.07 |

## Variant: does EI add value with U*? AUPRC of PRS* minus PD + U*

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | -0.037 ± 0.020 | 0 of 5 | 0.0625 | 0.125 | -2.33 |
| thyroid | 0.005 ± 0.006 | 5 of 5 | 0.0625 | 0.125 | 1.03 |

## PRS* minus the plain margin (AUPRC)

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | -0.016 ± 0.026 | 2 of 5 | 0.312 | 0.625 | -0.78 |
| thyroid | 0.001 ± 0.011 | 3 of 5 | 1 | 1 | 0.15 |

## PD + U* minus the plain margin (AUPRC)

Paired over seeds. Wilcoxon signed-rank test, Holm-corrected across datasets; d_z = mean difference / its standard deviation.

| Dataset | Difference | Seeds where first is higher | p | p (Holm) | d_z |
|---|---|---|---|---|---|
| S2 | 0.021 ± 0.011 | 5 of 5 | 0.0625 | 0.125 | 2.35 |
| thyroid | -0.004 ± 0.014 | 2 of 5 | 0.812 | 0.812 | -0.32 |

## Signal checks (gate G1)

Plausibility = share of perturbations whose L1 distance to the nearest training row is at most the 95th percentile of the original points' distances (near 0.95 or above = perturbations sit where real data sits). Spread = standard deviation over evaluation points.

| Dataset | Clipped coordinates | Plausibility | Spread of U | Spread of entropy | rho(PD, U) | rho(PD, EI) | rho(EI, U) | Signal time (s) |
|---|---|---|---|---|---|---|---|---|
| S2 | 0.012 | 0.95 | 0.039 | 0.002 | 0.62 | 0.01 | 0.09 | 62.2 |
| thyroid | 0.025 | 0.94 | 0.030 | 0.000 | 0.02 | 0.49 | -0.32 | 9.7 |
