# RQ1 faithfulness, fair protocol (v2)

50 test points per dataset and seed (fewer if the test split is smaller); seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. Features are ranked by signed attribution, most supportive of the predicted class first; KernelSHAP and LIME explain the predicted class's margin (KernelSHAP with 20 real background rows). Each deletion or insertion copies values from one random real background row (10 repetitions). Mean ± 95% CI over seeds.

Deletion margin AUC: lower = more faithful. Insertion margin AUC: higher = more faithful. '|attr| deletion' is the original ranking (secondary). Stability = top-3 overlap when the explainer is re-run with another seed. Ground truth = share of the top 3 (ties broken at random) that are truly relevant (synthetic sets only).

| Dataset | Method | Deletion margin AUC | Insertion margin AUC | |attr| deletion (old) | Class-kept deletion AUC | Stability | Ground truth top-3 | Negative attributions |
|---|---|---|---|---|---|---|---|---|
| S1 | PACE-contrastive | 0.049 ± 0.027 | 0.494 ± 0.028 | 0.116 ± 0.024 | 0.528 ± 0.015 | 1.00 | 0.46 | 0.28 |
| S1 | PACE-multi | -0.015 ± 0.023 | 0.546 ± 0.030 | 0.086 ± 0.021 | 0.494 ± 0.014 | 1.00 | 0.52 | 0.22 |
| S1 | PACE-flip | -0.088 ± 0.023 | 0.624 ± 0.018 | -0.088 ± 0.023 | 0.454 ± 0.014 | 1.00 | 0.58 | 0.00 |
| S1 | Occlusion (brute force) | -0.088 ± 0.023 | 0.624 ± 0.018 | -0.088 ± 0.023 | 0.454 ± 0.014 | 1.00 | 0.58 | 0.00 |
| S1 | PACE-decomposition | 0.148 ± 0.029 | 0.456 ± 0.030 | 0.159 ± 0.027 | 0.592 ± 0.016 | 1.00 | 0.33 | 0.08 |
| S1 | KernelSHAP | -0.128 ± 0.023 | 0.639 ± 0.017 | 0.061 ± 0.022 | 0.433 ± 0.013 | 0.73 | 0.82 | 0.40 |
| S1 | LIME | -0.115 ± 0.021 | 0.612 ± 0.018 | 0.073 ± 0.020 | 0.440 ± 0.012 | 0.89 | 0.92 | 0.42 |
| S1 | Random | 0.286 ± 0.017 | 0.279 ± 0.031 | 0.286 ± 0.017 | 0.666 ± 0.010 | 0.21 | 0.30 | 0.00 |
| S2 | PACE-contrastive | 0.030 ± 0.032 | 0.418 ± 0.039 | 0.137 ± 0.023 | 0.521 ± 0.019 | 1.00 | 0.42 | 0.29 |
| S2 | PACE-multi | -0.026 ± 0.021 | 0.480 ± 0.026 | 0.107 ± 0.023 | 0.490 ± 0.012 | 1.00 | 0.45 | 0.23 |
| S2 | PACE-flip | -0.117 ± 0.021 | 0.575 ± 0.020 | -0.117 ± 0.021 | 0.438 ± 0.012 | 1.00 | 0.51 | 0.00 |
| S2 | Occlusion (brute force) | -0.117 ± 0.021 | 0.575 ± 0.020 | -0.117 ± 0.021 | 0.438 ± 0.012 | 1.00 | 0.51 | 0.00 |
| S2 | PACE-decomposition | 0.131 ± 0.028 | 0.407 ± 0.020 | 0.146 ± 0.019 | 0.582 ± 0.016 | 1.00 | 0.32 | 0.08 |
| S2 | KernelSHAP | -0.159 ± 0.014 | 0.597 ± 0.019 | 0.070 ± 0.016 | 0.416 ± 0.008 | 0.65 | 0.76 | 0.37 |
| S2 | LIME | -0.131 ± 0.014 | 0.572 ± 0.022 | 0.091 ± 0.021 | 0.430 ± 0.007 | 0.87 | 0.91 | 0.42 |
| S2 | Random | 0.248 ± 0.032 | 0.248 ± 0.039 | 0.248 ± 0.032 | 0.643 ± 0.018 | 0.21 | 0.30 | 0.00 |
| S3 | PACE-contrastive | 0.131 ± 0.036 | 0.496 ± 0.027 | 0.191 ± 0.036 | 0.579 ± 0.020 | 1.00 | 0.47 | 0.28 |
| S3 | PACE-multi | 0.082 ± 0.039 | 0.547 ± 0.025 | 0.165 ± 0.042 | 0.551 ± 0.022 | 1.00 | 0.50 | 0.22 |
| S3 | PACE-flip | 0.005 ± 0.035 | 0.619 ± 0.018 | 0.005 ± 0.035 | 0.507 ± 0.021 | 1.00 | 0.54 | 0.00 |
| S3 | Occlusion (brute force) | 0.005 ± 0.035 | 0.619 ± 0.018 | 0.005 ± 0.035 | 0.507 ± 0.021 | 1.00 | 0.54 | 0.00 |
| S3 | PACE-decomposition | 0.225 ± 0.041 | 0.483 ± 0.023 | 0.234 ± 0.042 | 0.638 ± 0.022 | 1.00 | 0.32 | 0.07 |
| S3 | KernelSHAP | -0.035 ± 0.040 | 0.653 ± 0.014 | 0.154 ± 0.057 | 0.486 ± 0.023 | 0.66 | 0.77 | 0.41 |
| S3 | LIME | 0.021 ± 0.042 | 0.586 ± 0.025 | 0.167 ± 0.050 | 0.517 ± 0.023 | 0.86 | 0.91 | 0.43 |
| S3 | Random | 0.325 ± 0.040 | 0.337 ± 0.021 | 0.325 ± 0.040 | 0.693 ± 0.022 | 0.21 | 0.30 | 0.00 |
| wine | PACE-contrastive | 0.016 ± 0.034 | 0.429 ± 0.027 | 0.061 ± 0.032 | 0.517 ± 0.021 | 1.00 | – | 0.14 |
| wine | PACE-multi | -0.013 ± 0.031 | 0.466 ± 0.024 | 0.030 ± 0.030 | 0.505 ± 0.018 | 1.00 | – | 0.12 |
| wine | PACE-flip | -0.045 ± 0.031 | 0.505 ± 0.021 | -0.045 ± 0.031 | 0.487 ± 0.017 | 1.00 | – | 0.00 |
| wine | Occlusion (brute force) | -0.045 ± 0.031 | 0.505 ± 0.021 | -0.045 ± 0.031 | 0.487 ± 0.017 | 1.00 | – | 0.00 |
| wine | PACE-decomposition | 0.027 ± 0.029 | 0.477 ± 0.021 | 0.042 ± 0.031 | 0.522 ± 0.016 | 1.00 | – | 0.06 |
| wine | KernelSHAP | -0.068 ± 0.032 | 0.548 ± 0.022 | -0.045 ± 0.028 | 0.476 ± 0.016 | 0.77 | – | 0.04 |
| wine | LIME | -0.035 ± 0.032 | 0.511 ± 0.021 | 0.007 ± 0.030 | 0.493 ± 0.017 | 0.85 | – | 0.25 |
| wine | Random | 0.249 ± 0.016 | 0.248 ± 0.035 | 0.249 ± 0.016 | 0.645 ± 0.010 | 0.15 | – | 0.00 |
| breast_cancer | PACE-contrastive | 0.192 ± 0.024 | 0.627 ± 0.021 | 0.216 ± 0.026 | 0.625 ± 0.014 | 1.00 | – | 0.05 |
| breast_cancer | PACE-multi | 0.159 ± 0.026 | 0.689 ± 0.016 | 0.179 ± 0.029 | 0.606 ± 0.015 | 1.00 | – | 0.04 |
| breast_cancer | PACE-flip | 0.149 ± 0.022 | 0.708 ± 0.017 | 0.149 ± 0.022 | 0.605 ± 0.013 | 1.00 | – | 0.00 |
| breast_cancer | Occlusion (brute force) | 0.149 ± 0.022 | 0.708 ± 0.017 | 0.149 ± 0.022 | 0.605 ± 0.013 | 1.00 | – | 0.00 |
| breast_cancer | PACE-decomposition | 0.243 ± 0.024 | 0.643 ± 0.016 | 0.253 ± 0.025 | 0.654 ± 0.015 | 1.00 | – | 0.03 |
| breast_cancer | KernelSHAP | 0.193 ± 0.031 | 0.652 ± 0.021 | 0.227 ± 0.034 | 0.630 ± 0.017 | 0.45 | – | 0.05 |
| breast_cancer | LIME | 0.160 ± 0.024 | 0.700 ± 0.014 | 0.203 ± 0.026 | 0.605 ± 0.016 | 0.73 | – | 0.26 |
| breast_cancer | Random | 0.424 ± 0.021 | 0.432 ± 0.022 | 0.424 ± 0.021 | 0.765 ± 0.012 | 0.06 | – | 0.00 |
| thyroid | PACE-contrastive | 0.607 ± 0.046 | 0.858 ± 0.044 | 0.610 ± 0.047 | 0.902 ± 0.030 | 1.00 | – | 0.01 |
| thyroid | PACE-multi | 0.578 ± 0.052 | 0.901 ± 0.035 | 0.579 ± 0.054 | 0.893 ± 0.033 | 1.00 | – | 0.01 |
| thyroid | PACE-flip | 0.549 ± 0.054 | 0.917 ± 0.027 | 0.549 ± 0.054 | 0.880 ± 0.036 | 1.00 | – | 0.00 |
| thyroid | Occlusion (brute force) | 0.549 ± 0.054 | 0.917 ± 0.027 | 0.549 ± 0.054 | 0.879 ± 0.036 | 1.00 | – | 0.00 |
| thyroid | PACE-decomposition | 0.725 ± 0.055 | 0.747 ± 0.039 | 0.725 ± 0.055 | 0.916 ± 0.030 | 1.00 | – | 0.00 |
| thyroid | KernelSHAP | 0.577 ± 0.060 | 0.891 ± 0.025 | 0.628 ± 0.058 | 0.899 ± 0.035 | 0.46 | – | 0.31 |
| thyroid | LIME | 0.580 ± 0.050 | 0.881 ± 0.037 | 0.633 ± 0.049 | 0.898 ± 0.032 | 0.76 | – | 0.52 |
| thyroid | Random | 0.742 ± 0.046 | 0.737 ± 0.047 | 0.742 ± 0.046 | 0.939 ± 0.021 | 0.09 | – | 0.00 |
| shuttle | PACE-contrastive | 0.362 ± 0.083 | 0.647 ± 0.061 | 0.361 ± 0.082 | 0.749 ± 0.040 | 1.00 | – | 0.05 |
| shuttle | PACE-multi | 0.344 ± 0.072 | 0.683 ± 0.060 | 0.343 ± 0.072 | 0.723 ± 0.032 | 1.00 | – | 0.04 |
| shuttle | PACE-flip | 0.344 ± 0.077 | 0.724 ± 0.051 | 0.344 ± 0.077 | 0.711 ± 0.036 | 1.00 | – | 0.00 |
| shuttle | Occlusion (brute force) | 0.344 ± 0.077 | 0.724 ± 0.051 | 0.344 ± 0.077 | 0.711 ± 0.036 | 1.00 | – | 0.00 |
| shuttle | PACE-decomposition | 0.363 ± 0.073 | 0.695 ± 0.053 | 0.363 ± 0.073 | 0.707 ± 0.034 | 1.00 | – | 0.00 |
| shuttle | KernelSHAP | 0.331 ± 0.072 | 0.712 ± 0.057 | 0.344 ± 0.076 | 0.705 ± 0.038 | 0.55 | – | 0.22 |
| shuttle | LIME | 0.346 ± 0.074 | 0.693 ± 0.054 | 0.349 ± 0.078 | 0.717 ± 0.034 | 0.89 | – | 0.39 |
| shuttle | Random | 0.507 ± 0.060 | 0.520 ± 0.072 | 0.507 ± 0.060 | 0.801 ± 0.024 | 0.22 | – | 0.00 |
| magic | PACE-contrastive | 0.087 ± 0.022 | 0.129 ± 0.031 | 0.090 ± 0.013 | 0.568 ± 0.023 | 1.00 | – | 0.32 |
| magic | PACE-multi | 0.067 ± 0.012 | 0.179 ± 0.029 | 0.069 ± 0.010 | 0.557 ± 0.019 | 1.00 | – | 0.17 |
| magic | PACE-flip | 0.029 ± 0.021 | 0.236 ± 0.019 | 0.029 ± 0.021 | 0.506 ± 0.016 | 1.00 | – | 0.00 |
| magic | Occlusion (brute force) | 0.029 ± 0.021 | 0.236 ± 0.019 | 0.029 ± 0.021 | 0.506 ± 0.016 | 1.00 | – | 0.00 |
| magic | PACE-decomposition | 0.089 ± 0.015 | 0.148 ± 0.019 | 0.089 ± 0.015 | 0.584 ± 0.013 | 1.00 | – | 0.00 |
| magic | KernelSHAP | -0.021 ± 0.022 | 0.268 ± 0.015 | 0.083 ± 0.014 | 0.522 ± 0.017 | 0.42 | – | 0.45 |
| magic | LIME | -0.001 ± 0.022 | 0.224 ± 0.027 | 0.068 ± 0.015 | 0.509 ± 0.019 | 0.68 | – | 0.42 |
| magic | Random | 0.138 ± 0.025 | 0.102 ± 0.022 | 0.138 ± 0.025 | 0.634 ± 0.017 | 0.21 | – | 0.00 |
| letter | PACE-contrastive | -0.461 ± 0.030 | -0.386 ± 0.029 | -0.446 ± 0.028 | 0.254 ± 0.016 | 1.00 | – | 0.22 |
| letter | PACE-multi | -0.562 ± 0.022 | -0.256 ± 0.033 | -0.550 ± 0.021 | 0.185 ± 0.015 | 1.00 | – | 0.13 |
| letter | PACE-flip | -0.656 ± 0.018 | -0.056 ± 0.037 | -0.656 ± 0.018 | 0.127 ± 0.009 | 1.00 | – | 0.00 |
| letter | Occlusion (brute force) | -0.656 ± 0.018 | -0.056 ± 0.037 | -0.656 ± 0.018 | 0.127 ± 0.009 | 1.00 | – | 0.00 |
| letter | PACE-decomposition | -0.611 ± 0.027 | -0.128 ± 0.026 | -0.610 ± 0.027 | 0.175 ± 0.012 | 1.00 | – | 0.00 |
| letter | KernelSHAP | -0.690 ± 0.014 | -0.025 ± 0.035 | -0.688 ± 0.014 | 0.132 ± 0.007 | 0.61 | – | 0.00 |
| letter | LIME | -0.663 ± 0.016 | -0.114 ± 0.037 | -0.624 ± 0.015 | 0.136 ± 0.010 | 0.85 | – | 0.36 |
| letter | Random | -0.399 ± 0.036 | -0.413 ± 0.032 | -0.399 ± 0.036 | 0.305 ± 0.020 | 0.12 | – | 0.00 |
| landsat | PACE-contrastive | -0.143 ± 0.029 | 0.032 ± 0.028 | -0.138 ± 0.024 | 0.436 ± 0.017 | 1.00 | – | 0.10 |
| landsat | PACE-multi | -0.213 ± 0.029 | 0.082 ± 0.028 | -0.209 ± 0.029 | 0.398 ± 0.018 | 1.00 | – | 0.06 |
| landsat | PACE-flip | -0.260 ± 0.028 | 0.143 ± 0.030 | -0.260 ± 0.028 | 0.374 ± 0.017 | 1.00 | – | 0.00 |
| landsat | Occlusion (brute force) | -0.260 ± 0.028 | 0.143 ± 0.030 | -0.260 ± 0.028 | 0.374 ± 0.017 | 1.00 | – | 0.00 |
| landsat | PACE-decomposition | -0.182 ± 0.035 | 0.151 ± 0.027 | -0.182 ± 0.035 | 0.419 ± 0.017 | 1.00 | – | 0.00 |
| landsat | KernelSHAP | -0.233 ± 0.037 | 0.119 ± 0.031 | -0.232 ± 0.037 | 0.398 ± 0.020 | 0.19 | – | 0.00 |
| landsat | LIME | -0.296 ± 0.036 | 0.142 ± 0.034 | -0.272 ± 0.032 | 0.356 ± 0.017 | 0.48 | – | 0.26 |
| landsat | Random | -0.046 ± 0.040 | -0.045 ± 0.032 | -0.046 ± 0.040 | 0.496 ± 0.020 | 0.05 | – | 0.00 |
| page_blocks | PACE-contrastive | 0.104 ± 0.055 | 0.162 ± 0.071 | 0.119 ± 0.059 | 0.717 ± 0.060 | 1.00 | – | 0.27 |
| page_blocks | PACE-multi | 0.072 ± 0.047 | 0.258 ± 0.099 | 0.079 ± 0.052 | 0.709 ± 0.052 | 1.00 | – | 0.12 |
| page_blocks | PACE-flip | 0.064 ± 0.055 | 0.267 ± 0.083 | 0.064 ± 0.055 | 0.695 ± 0.057 | 1.00 | – | 0.00 |
| page_blocks | Occlusion (brute force) | 0.064 ± 0.055 | 0.267 ± 0.084 | 0.064 ± 0.055 | 0.695 ± 0.057 | 1.00 | – | 0.00 |
| page_blocks | PACE-decomposition | 0.145 ± 0.080 | 0.159 ± 0.059 | 0.146 ± 0.080 | 0.774 ± 0.056 | 1.00 | – | 0.00 |
| page_blocks | KernelSHAP | 0.055 ± 0.052 | 0.270 ± 0.088 | 0.117 ± 0.059 | 0.749 ± 0.057 | 0.46 | – | 0.50 |
| page_blocks | LIME | 0.068 ± 0.055 | 0.260 ± 0.085 | 0.131 ± 0.066 | 0.739 ± 0.055 | 0.83 | – | 0.54 |
| page_blocks | Random | 0.178 ± 0.064 | 0.135 ± 0.075 | 0.178 ± 0.064 | 0.800 ± 0.036 | 0.21 | – | 0.00 |

## PACE-flip minus KernelSHAP: deletion margin AUC (lower is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: negative = PACE-flip better (more negative).

| Dataset | Difference | Seeds where PACE-flip is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.040 ± 0.005 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.041 ± 0.011 | 10 of 10 | 0.00195 | 0.0215 |
| S3 | +0.041 ± 0.007 | 10 of 10 | 0.00195 | 0.0215 |
| wine | +0.023 ± 0.009 | 10 of 10 | 0.00195 | 0.0215 |
| breast_cancer | -0.045 ± 0.015 | 0 of 10 | 0.00195 | 0.0215 |
| thyroid | -0.028 ± 0.019 | 1 of 10 | 0.00391 | 0.0215 |
| shuttle | +0.013 ± 0.019 | 5 of 10 | 0.375 | 0.375 |
| magic | +0.050 ± 0.016 | 10 of 10 | 0.00195 | 0.0215 |
| letter | +0.034 ± 0.010 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | -0.027 ± 0.018 | 0 of 10 | 0.00195 | 0.0215 |
| page_blocks | +0.008 ± 0.008 | 8 of 10 | 0.0371 | 0.0742 |

## PACE-flip minus LIME: deletion margin AUC (lower is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: negative = PACE-flip better (more negative).

| Dataset | Difference | Seeds where PACE-flip is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.027 ± 0.007 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.013 ± 0.009 | 9 of 10 | 0.0195 | 0.117 |
| S3 | -0.015 ± 0.017 | 3 of 10 | 0.105 | 0.527 |
| wine | -0.010 ± 0.012 | 3 of 10 | 0.105 | 0.527 |
| breast_cancer | -0.011 ± 0.006 | 1 of 10 | 0.00586 | 0.0469 |
| thyroid | -0.031 ± 0.011 | 1 of 10 | 0.00391 | 0.0352 |
| shuttle | -0.002 ± 0.017 | 6 of 10 | 1 | 1 |
| magic | +0.030 ± 0.015 | 10 of 10 | 0.00195 | 0.0215 |
| letter | +0.007 ± 0.013 | 8 of 10 | 0.193 | 0.58 |
| landsat | +0.036 ± 0.020 | 9 of 10 | 0.00586 | 0.0469 |
| page_blocks | -0.005 ± 0.011 | 4 of 10 | 0.432 | 0.863 |

## PACE-flip minus PACE-multi: deletion margin AUC (lower is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: negative = PACE-flip better (more negative).

| Dataset | Difference | Seeds where PACE-flip is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.074 ± 0.010 | 0 of 10 | 0.00195 | 0.0215 |
| S2 | -0.092 ± 0.014 | 0 of 10 | 0.00195 | 0.0215 |
| S3 | -0.076 ± 0.017 | 0 of 10 | 0.00195 | 0.0215 |
| wine | -0.033 ± 0.008 | 0 of 10 | 0.00195 | 0.0215 |
| breast_cancer | -0.010 ± 0.010 | 2 of 10 | 0.0645 | 0.193 |
| thyroid | -0.029 ± 0.013 | 1 of 10 | 0.00391 | 0.0215 |
| shuttle | +0.000 ± 0.014 | 5 of 10 | 1 | 1 |
| magic | -0.038 ± 0.016 | 0 of 10 | 0.00195 | 0.0215 |
| letter | -0.094 ± 0.013 | 0 of 10 | 0.00195 | 0.0215 |
| landsat | -0.046 ± 0.012 | 0 of 10 | 0.00195 | 0.0215 |
| page_blocks | -0.009 ± 0.013 | 3 of 10 | 0.232 | 0.465 |

## PACE-flip minus KernelSHAP: insertion margin AUC (higher is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: positive = PACE-flip better.

| Dataset | Difference | Seeds where PACE-flip is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.015 ± 0.008 | 1 of 10 | 0.00391 | 0.0215 |
| S2 | -0.021 ± 0.006 | 0 of 10 | 0.00195 | 0.0215 |
| S3 | -0.034 ± 0.010 | 0 of 10 | 0.00195 | 0.0215 |
| wine | -0.044 ± 0.011 | 0 of 10 | 0.00195 | 0.0215 |
| breast_cancer | +0.056 ± 0.011 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.026 ± 0.012 | 10 of 10 | 0.00195 | 0.0215 |
| shuttle | +0.012 ± 0.018 | 7 of 10 | 0.193 | 0.387 |
| magic | -0.032 ± 0.014 | 0 of 10 | 0.00195 | 0.0215 |
| letter | -0.031 ± 0.013 | 0 of 10 | 0.00195 | 0.0215 |
| landsat | +0.024 ± 0.020 | 9 of 10 | 0.0488 | 0.146 |
| page_blocks | -0.004 ± 0.020 | 4 of 10 | 0.846 | 0.846 |

## PACE-flip minus LIME: insertion margin AUC (higher is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: positive = PACE-flip better.

| Dataset | Difference | Seeds where PACE-flip is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.011 ± 0.010 | 9 of 10 | 0.00586 | 0.041 |
| S2 | +0.003 ± 0.007 | 5 of 10 | 0.432 | 1 |
| S3 | +0.034 ± 0.015 | 9 of 10 | 0.00391 | 0.0312 |
| wine | -0.006 ± 0.007 | 3 of 10 | 0.084 | 0.42 |
| breast_cancer | +0.008 ± 0.009 | 8 of 10 | 0.0645 | 0.387 |
| thyroid | +0.036 ± 0.013 | 10 of 10 | 0.00195 | 0.0215 |
| shuttle | +0.031 ± 0.015 | 10 of 10 | 0.00195 | 0.0215 |
| magic | +0.012 ± 0.021 | 6 of 10 | 0.322 | 1 |
| letter | +0.058 ± 0.010 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.001 ± 0.021 | 5 of 10 | 0.846 | 1 |
| page_blocks | +0.007 ± 0.026 | 6 of 10 | 0.77 | 1 |

## PACE-flip minus PACE-multi: insertion margin AUC (higher is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: positive = PACE-flip better.

| Dataset | Difference | Seeds where PACE-flip is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.078 ± 0.017 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.096 ± 0.013 | 10 of 10 | 0.00195 | 0.0215 |
| S3 | +0.073 ± 0.015 | 10 of 10 | 0.00195 | 0.0215 |
| wine | +0.038 ± 0.006 | 10 of 10 | 0.00195 | 0.0215 |
| breast_cancer | +0.018 ± 0.005 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.016 ± 0.011 | 8 of 10 | 0.0137 | 0.0273 |
| shuttle | +0.041 ± 0.019 | 9 of 10 | 0.00586 | 0.0215 |
| magic | +0.057 ± 0.019 | 10 of 10 | 0.00195 | 0.0215 |
| letter | +0.200 ± 0.012 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.061 ± 0.009 | 10 of 10 | 0.00195 | 0.0215 |
| page_blocks | +0.009 ± 0.024 | 7 of 10 | 0.375 | 0.375 |

## The model's own decision sensitivity (synthetic sets)

Share of all single-feature decision flips (PACE-flip's flip mass, computed from the model) that come from the truly relevant features 0–2, and share of points where replacing some irrelevant feature can flip the decision.

| Dataset | Flip mass on relevant features | Points with an irrelevant feature that flips |
|---|---|---|
| S1 | 0.489 ± 0.030 | 0.684 ± 0.047 |
| S2 | 0.413 ± 0.024 | 0.808 ± 0.028 |
| S3 | 0.477 ± 0.039 | 0.730 ± 0.021 |
