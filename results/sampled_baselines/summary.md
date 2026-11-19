# Run J: sampled baselines above 12 features, at an equal evaluation budget

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's model, split, points and rows; seeds 160–169 (10). Each comparison: paired two-sided Wilcoxon over seeds, Holm across the datasets; better / level / worse for PACE-SHAP. Written by `scripts/run_sampled_baselines.py`.

## The budget

| Dataset | Features | Background rows | Points per seed | Boxes (mean) | Budget: model rows per explanation | Rows used: PACE-SHAP | Rows used: PermutationSHAP | Rows used: KernelSHAP-paired |
|---|---|---|---|---|---|---|---|---|
| wine | 13 | 71 | 36 | 31 | 31,808 | 31,808 | 30,302 | 31,738 |
| breast_cancer | 30 | 200 | 50 | 44 | 198,400 | 198,400 | 195,116 | 198,201 |
| thyroid | 21 | 200 | 50 | 51 | 140,800 | 140,800 | 53,501 | 140,601 |
| letter | 16 | 200 | 50 | 556 | 108,800 | 108,800 | 90,300 | 108,601 |
| landsat | 36 | 200 | 50 | 200 | 236,800 | 236,800 | 227,603 | 236,601 |

PACE-SHAP's rows are the coalition values its orders walk through (it reads them from the model's terms and does not call the model); the baselines' rows were counted as they called the model.

## Primary: deletion margin AUC (lower is more faithful)

**PACE-SHAP vs PermutationSHAP**: better 1, level 4, worse 0

| Dataset | Mean difference (PACE-SHAP − PermutationSHAP) | Seeds in favour of PACE-SHAP | p | Holm p | Verdict |
|---|---|---|---|---|---|
| wine | -0.0004 | 6 of 10 | 0.2754 | 0.8262 | level |
| breast_cancer | -0.0004 | 8 of 10 | 0.0371 | 0.1484 | level |
| thyroid | -0.0006 | 5 of 10 | 0.2754 | 0.8262 | level |
| letter | -0.0004 | 6 of 10 | 0.6953 | 0.8262 | level |
| landsat | -0.0045 | 10 of 10 | 0.0020 | 0.0098 | better |

**PACE-SHAP vs KernelSHAP-paired**: better 1, level 4, worse 0

| Dataset | Mean difference (PACE-SHAP − KernelSHAP-paired) | Seeds in favour of PACE-SHAP | p | Holm p | Verdict |
|---|---|---|---|---|---|
| wine | -0.0003 | 8 of 10 | 0.1055 | 0.2109 | level |
| breast_cancer | -0.0005 | 8 of 10 | 0.0371 | 0.1484 | level |
| thyroid | -0.0005 | 8 of 10 | 0.0488 | 0.1484 | level |
| letter | -0.0004 | 7 of 10 | 0.5566 | 0.5566 | level |
| landsat | -0.0057 | 10 of 10 | 0.0020 | 0.0098 | better |

Mean deletion margin AUC over seeds:

| Dataset | PACE-SHAP | PermutationSHAP | KernelSHAP-paired | Random |
|---|---|---|---|---|
| wine | -0.0748 | -0.0744 | -0.0745 | 0.2358 |
| breast_cancer | 0.1299 | 0.1303 | 0.1305 | 0.4094 |
| thyroid | 0.5372 | 0.5378 | 0.5377 | 0.7420 |
| letter | -0.6809 | -0.6805 | -0.6806 | -0.3764 |
| landsat | -0.3422 | -0.3377 | -0.3365 | -0.0419 |

## Secondary: insertion margin AUC (higher is more faithful)

**PACE-SHAP vs PermutationSHAP**: better 1, level 4, worse 0

| Dataset | Mean difference (PACE-SHAP − PermutationSHAP) | Seeds in favour of PACE-SHAP | p | Holm p | Verdict |
|---|---|---|---|---|---|
| wine | +0.0011 | 10 of 10 | 0.0020 | 0.0098 | better |
| breast_cancer | +0.0003 | 6 of 10 | 0.2324 | 0.6973 | level |
| thyroid | +0.0004 | 5 of 10 | 0.8457 | 1.0000 | level |
| letter | -0.0006 | 4 of 10 | 0.6250 | 1.0000 | level |
| landsat | +0.0022 | 7 of 10 | 0.1055 | 0.4219 | level |

**PACE-SHAP vs KernelSHAP-paired**: better 2, level 3, worse 0

| Dataset | Mean difference (PACE-SHAP − KernelSHAP-paired) | Seeds in favour of PACE-SHAP | p | Holm p | Verdict |
|---|---|---|---|---|---|
| wine | +0.0014 | 9 of 10 | 0.0195 | 0.0586 | level |
| breast_cancer | +0.0001 | 5 of 10 | 0.9219 | 0.9219 | level |
| thyroid | +0.0008 | 10 of 10 | 0.0020 | 0.0098 | better |
| letter | -0.0014 | 3 of 10 | 0.2754 | 0.5508 | level |
| landsat | +0.0056 | 8 of 10 | 0.0098 | 0.0391 | better |

Mean insertion margin AUC over seeds:

| Dataset | PACE-SHAP | PermutationSHAP | KernelSHAP-paired | Random |
|---|---|---|---|---|
| wine | 0.5364 | 0.5352 | 0.5349 | 0.2380 |
| breast_cancer | 0.7048 | 0.7045 | 0.7047 | 0.4228 |
| thyroid | 0.9167 | 0.9163 | 0.9159 | 0.7247 |
| letter | 0.0272 | 0.0278 | 0.0285 | -0.4012 |
| landsat | 0.2492 | 0.2470 | 0.2436 | -0.0362 |

## Error against exact values (d ≤ 16; the first 10 points of each seed)

Relative L1 error: the sum over features of |estimate − exact|, divided by the sum of |exact|, averaged over points; mean over seeds.

| Dataset | PACE-SHAP | PermutationSHAP | KernelSHAP-paired |
|---|---|---|---|
| wine | 0.0100 | 0.0288 | 0.0252 |
| letter | 0.0139 | 0.0458 | 0.0490 |

Largest absolute error (mean over seeds):

| Dataset | PACE-SHAP | PermutationSHAP | KernelSHAP-paired |
|---|---|---|---|
| wine | 0.00100 | 0.00385 | 0.00273 |
| letter | 0.00042 | 0.00210 | 0.00223 |

Top-3 overlap with the exact values (Jaccard, by absolute value):

| Dataset | PACE-SHAP | PermutationSHAP | KernelSHAP-paired |
|---|---|---|---|
| wine | 0.980 | 0.970 | 0.985 |
| letter | 0.980 | 0.955 | 0.945 |

## Top-3 stability under a new seed (Jaccard overlap, mean over seeds)

| Dataset | PACE-SHAP | PermutationSHAP | KernelSHAP-paired | Random |
|---|---|---|---|---|
| wine | 0.981 | 0.947 | 0.972 | 0.153 |
| breast_cancer | 0.974 | 0.888 | 0.906 | 0.059 |
| thyroid | 0.991 | 0.971 | 0.986 | 0.089 |
| letter | 0.974 | 0.912 | 0.924 | 0.129 |
| landsat | 0.950 | 0.761 | 0.714 | 0.047 |

## Milliseconds per explanation during the run

Several tasks shared the CPU, so compare within a dataset only. The times measured alone are below, if `--timing` has been run.

| Dataset | PACE-SHAP | PermutationSHAP | KernelSHAP-paired |
|---|---|---|---|
| wine | 16 | 124 | 122 |
| breast_cancer | 253 | 1,897 | 1,839 |
| thyroid | 227 | 542 | 1,331 |
| letter | 1,442 | 5,824 | 5,334 |
| landsat | 3,086 | 10,111 | 10,761 |

## Milliseconds per explanation, measured alone (seeds [160, 161, 162])

| Dataset | PACE-SHAP | PermutationSHAP | KernelSHAP-paired |
|---|---|---|---|
| wine | 7 | 69 | 58 |
| breast_cancer | 84 | 805 | 802 |
| thyroid | 60 | 228 | 562 |
| letter | 524 | 2,550 | 3,069 |
| landsat | 636 | 4,451 | 4,594 |

Times PACE-SHAP's:

| Dataset | PermutationSHAP | KernelSHAP-paired |
|---|---|---|
| wine | 9.4× | 7.8× |
| breast_cancer | 9.6× | 9.6× |
| thyroid | 3.8× | 9.3× |
| letter | 4.9× | 5.9× |
| landsat | 7.0× | 7.2× |

