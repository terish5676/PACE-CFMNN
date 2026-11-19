# RQ1 faithfulness, v3 (PACE-SHAP; separate explainer background)

50 test points per dataset and seed (fewer if the test split is smaller); seeds [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]. Same tests as v2 (signed ranking, margin targets), but the deletion and insertion tests copy values from one set of training rows and every explainer uses a second, disjoint set (200 rows each, or half the training set). PACE-SHAP uses the first 200 explainer rows (exact for d <= 12, otherwise 32 orders per row); PACE-SHAP-20 uses the same 20 rows as KernelSHAP; KernelSHAP-strong uses 2048 samples and no l1 feature selection (every coalition when d <= 11; no stability re-run). Each deletion or insertion copies values from one random evaluation row (10 repetitions). Mean ± 95% CI over seeds.

Deletion margin AUC: lower = more faithful. Insertion margin AUC: higher = more faithful. '|attr| deletion' is the original ranking (secondary). Stability = top-3 overlap when the explainer is re-run with another seed. Ground truth = share of the top 3 (ties broken at random) that are truly relevant (synthetic sets only).

| Dataset | Method | Deletion margin AUC | Insertion margin AUC | |attr| deletion (old) | Class-kept deletion AUC | Stability | Ground truth top-3 | Negative attributions |
|---|---|---|---|---|---|---|---|---|
| S1 | PACE-SHAP | -0.139 ± 0.035 | 0.645 ± 0.015 | 0.062 ± 0.016 | 0.428 ± 0.018 | 1.00 | 0.85 | 0.38 |
| S1 | PACE-SHAP-20 | -0.131 ± 0.030 | 0.636 ± 0.014 | 0.078 ± 0.019 | 0.433 ± 0.015 | 0.74 | 0.81 | 0.39 |
| S1 | PACE-flip | -0.089 ± 0.032 | 0.617 ± 0.012 | -0.089 ± 0.032 | 0.456 ± 0.016 | 1.00 | 0.55 | 0.00 |
| S1 | PACE-multi | -0.016 ± 0.037 | 0.542 ± 0.021 | 0.095 ± 0.024 | 0.495 ± 0.019 | 1.00 | 0.51 | 0.23 |
| S1 | PACE-contrastive | 0.047 ± 0.037 | 0.483 ± 0.021 | 0.133 ± 0.030 | 0.531 ± 0.020 | 1.00 | 0.47 | 0.29 |
| S1 | PACE-decomposition | 0.154 ± 0.027 | 0.436 ± 0.017 | 0.167 ± 0.021 | 0.596 ± 0.014 | 1.00 | 0.32 | 0.08 |
| S1 | KernelSHAP | -0.133 ± 0.031 | 0.635 ± 0.013 | 0.076 ± 0.018 | 0.431 ± 0.015 | 0.73 | 0.81 | 0.39 |
| S1 | KernelSHAP-strong | -0.131 ± 0.030 | 0.636 ± 0.014 | 0.078 ± 0.019 | 0.433 ± 0.015 | – | 0.81 | 0.39 |
| S1 | LIME | -0.109 ± 0.037 | 0.609 ± 0.016 | 0.086 ± 0.016 | 0.443 ± 0.018 | 0.90 | 0.92 | 0.42 |
| S1 | Random | 0.271 ± 0.033 | 0.296 ± 0.024 | 0.271 ± 0.033 | 0.659 ± 0.018 | 0.21 | 0.29 | 0.00 |
| S2 | PACE-SHAP | -0.132 ± 0.034 | 0.609 ± 0.020 | 0.079 ± 0.024 | 0.432 ± 0.018 | 1.00 | 0.79 | 0.37 |
| S2 | PACE-SHAP-20 | -0.128 ± 0.032 | 0.593 ± 0.021 | 0.081 ± 0.018 | 0.434 ± 0.018 | 0.67 | 0.76 | 0.38 |
| S2 | PACE-flip | -0.076 ± 0.037 | 0.561 ± 0.020 | -0.076 ± 0.037 | 0.464 ± 0.020 | 1.00 | 0.49 | 0.00 |
| S2 | PACE-multi | -0.008 ± 0.035 | 0.480 ± 0.024 | 0.099 ± 0.018 | 0.500 ± 0.019 | 1.00 | 0.44 | 0.23 |
| S2 | PACE-contrastive | 0.054 ± 0.032 | 0.420 ± 0.019 | 0.147 ± 0.020 | 0.534 ± 0.017 | 1.00 | 0.40 | 0.28 |
| S2 | PACE-decomposition | 0.149 ± 0.038 | 0.405 ± 0.023 | 0.156 ± 0.037 | 0.590 ± 0.019 | 1.00 | 0.31 | 0.08 |
| S2 | KernelSHAP | -0.125 ± 0.032 | 0.592 ± 0.021 | 0.086 ± 0.017 | 0.437 ± 0.017 | 0.67 | 0.75 | 0.38 |
| S2 | KernelSHAP-strong | -0.128 ± 0.032 | 0.593 ± 0.021 | 0.081 ± 0.018 | 0.434 ± 0.018 | – | 0.76 | 0.38 |
| S2 | LIME | -0.095 ± 0.036 | 0.565 ± 0.027 | 0.104 ± 0.020 | 0.452 ± 0.020 | 0.88 | 0.91 | 0.43 |
| S2 | Random | 0.256 ± 0.047 | 0.257 ± 0.037 | 0.256 ± 0.047 | 0.650 ± 0.025 | 0.21 | 0.29 | 0.00 |
| S3 | PACE-SHAP | -0.055 ± 0.051 | 0.639 ± 0.021 | 0.138 ± 0.037 | 0.478 ± 0.030 | 1.00 | 0.80 | 0.42 |
| S3 | PACE-SHAP-20 | -0.044 ± 0.050 | 0.627 ± 0.021 | 0.149 ± 0.047 | 0.483 ± 0.028 | 0.69 | 0.72 | 0.43 |
| S3 | PACE-flip | 0.006 ± 0.050 | 0.595 ± 0.022 | 0.006 ± 0.050 | 0.508 ± 0.030 | 1.00 | 0.52 | 0.00 |
| S3 | PACE-multi | 0.074 ± 0.050 | 0.520 ± 0.033 | 0.156 ± 0.036 | 0.547 ± 0.029 | 1.00 | 0.48 | 0.23 |
| S3 | PACE-contrastive | 0.119 ± 0.046 | 0.474 ± 0.036 | 0.184 ± 0.033 | 0.573 ± 0.027 | 1.00 | 0.44 | 0.28 |
| S3 | PACE-decomposition | 0.219 ± 0.053 | 0.451 ± 0.033 | 0.229 ± 0.051 | 0.634 ± 0.031 | 1.00 | 0.36 | 0.08 |
| S3 | KernelSHAP | -0.041 ± 0.048 | 0.626 ± 0.021 | 0.150 ± 0.045 | 0.485 ± 0.028 | 0.65 | 0.72 | 0.43 |
| S3 | KernelSHAP-strong | -0.044 ± 0.050 | 0.627 ± 0.021 | 0.149 ± 0.047 | 0.483 ± 0.028 | – | 0.72 | 0.43 |
| S3 | LIME | 0.012 ± 0.050 | 0.569 ± 0.021 | 0.167 ± 0.037 | 0.513 ± 0.030 | 0.85 | 0.89 | 0.43 |
| S3 | Random | 0.318 ± 0.046 | 0.314 ± 0.037 | 0.318 ± 0.046 | 0.687 ± 0.025 | 0.21 | 0.29 | 0.00 |
| wine | PACE-SHAP | -0.076 ± 0.036 | 0.533 ± 0.019 | -0.051 ± 0.033 | 0.472 ± 0.020 | 0.97 | – | 0.12 |
| wine | PACE-SHAP-20 | -0.070 ± 0.038 | 0.526 ± 0.018 | -0.043 ± 0.034 | 0.476 ± 0.022 | 0.71 | – | 0.13 |
| wine | PACE-flip | -0.055 ± 0.030 | 0.489 ± 0.018 | -0.055 ± 0.030 | 0.480 ± 0.018 | 1.00 | – | 0.00 |
| wine | PACE-multi | -0.025 ± 0.032 | 0.448 ± 0.020 | 0.013 ± 0.031 | 0.498 ± 0.019 | 1.00 | – | 0.12 |
| wine | PACE-contrastive | 0.005 ± 0.029 | 0.409 ± 0.022 | 0.046 ± 0.029 | 0.512 ± 0.018 | 1.00 | – | 0.14 |
| wine | PACE-decomposition | 0.040 ± 0.039 | 0.446 ± 0.024 | 0.056 ± 0.037 | 0.531 ± 0.022 | 1.00 | – | 0.07 |
| wine | KernelSHAP | -0.068 ± 0.038 | 0.524 ± 0.018 | -0.046 ± 0.034 | 0.476 ± 0.022 | 0.71 | – | 0.04 |
| wine | KernelSHAP-strong | -0.070 ± 0.039 | 0.526 ± 0.018 | -0.043 ± 0.034 | 0.476 ± 0.022 | – | – | 0.13 |
| wine | LIME | -0.039 ± 0.035 | 0.492 ± 0.018 | 0.002 ± 0.031 | 0.491 ± 0.019 | 0.89 | – | 0.26 |
| wine | Random | 0.236 ± 0.038 | 0.239 ± 0.021 | 0.236 ± 0.038 | 0.641 ± 0.022 | 0.15 | – | 0.00 |
| breast_cancer | PACE-SHAP | 0.111 ± 0.021 | 0.699 ± 0.019 | 0.173 ± 0.022 | 0.591 ± 0.013 | 0.97 | – | 0.31 |
| breast_cancer | PACE-SHAP-20 | 0.122 ± 0.020 | 0.687 ± 0.023 | 0.190 ± 0.026 | 0.597 ± 0.013 | 0.55 | – | 0.33 |
| breast_cancer | PACE-flip | 0.137 ± 0.022 | 0.685 ± 0.020 | 0.137 ± 0.022 | 0.606 ± 0.013 | 1.00 | – | 0.00 |
| breast_cancer | PACE-multi | 0.143 ± 0.020 | 0.668 ± 0.021 | 0.160 ± 0.020 | 0.608 ± 0.015 | 1.00 | – | 0.05 |
| breast_cancer | PACE-contrastive | 0.179 ± 0.026 | 0.598 ± 0.029 | 0.199 ± 0.025 | 0.625 ± 0.016 | 1.00 | – | 0.05 |
| breast_cancer | PACE-decomposition | 0.222 ± 0.024 | 0.622 ± 0.020 | 0.230 ± 0.025 | 0.643 ± 0.014 | 1.00 | – | 0.03 |
| breast_cancer | KernelSHAP | 0.174 ± 0.023 | 0.634 ± 0.024 | 0.207 ± 0.028 | 0.626 ± 0.012 | 0.38 | – | 0.04 |
| breast_cancer | KernelSHAP-strong | 0.122 ± 0.020 | 0.687 ± 0.023 | 0.190 ± 0.025 | 0.598 ± 0.013 | – | – | 0.33 |
| breast_cancer | LIME | 0.141 ± 0.020 | 0.680 ± 0.018 | 0.184 ± 0.022 | 0.600 ± 0.013 | 0.72 | – | 0.27 |
| breast_cancer | Random | 0.398 ± 0.015 | 0.414 ± 0.015 | 0.398 ± 0.015 | 0.752 ± 0.009 | 0.07 | – | 0.00 |
| thyroid | PACE-SHAP | 0.549 ± 0.047 | 0.915 ± 0.031 | 0.625 ± 0.050 | 0.889 ± 0.028 | 0.99 | – | 0.69 |
| thyroid | PACE-SHAP-20 | 0.562 ± 0.049 | 0.902 ± 0.027 | 0.624 ± 0.049 | 0.898 ± 0.031 | 0.55 | – | 0.41 |
| thyroid | PACE-flip | 0.544 ± 0.047 | 0.918 ± 0.027 | 0.544 ± 0.047 | 0.879 ± 0.031 | 1.00 | – | 0.00 |
| thyroid | PACE-multi | 0.582 ± 0.048 | 0.904 ± 0.027 | 0.584 ± 0.049 | 0.891 ± 0.031 | 1.00 | – | 0.01 |
| thyroid | PACE-contrastive | 0.599 ± 0.046 | 0.856 ± 0.037 | 0.603 ± 0.046 | 0.897 ± 0.029 | 1.00 | – | 0.01 |
| thyroid | PACE-decomposition | 0.723 ± 0.046 | 0.743 ± 0.032 | 0.724 ± 0.046 | 0.916 ± 0.026 | 1.00 | – | 0.00 |
| thyroid | KernelSHAP | 0.575 ± 0.043 | 0.892 ± 0.032 | 0.638 ± 0.047 | 0.901 ± 0.028 | 0.55 | – | 0.36 |
| thyroid | KernelSHAP-strong | 0.562 ± 0.048 | 0.903 ± 0.028 | 0.624 ± 0.049 | 0.898 ± 0.031 | – | – | 0.41 |
| thyroid | LIME | 0.571 ± 0.046 | 0.874 ± 0.027 | 0.640 ± 0.049 | 0.897 ± 0.029 | 0.77 | – | 0.50 |
| thyroid | Random | 0.739 ± 0.043 | 0.736 ± 0.031 | 0.739 ± 0.043 | 0.938 ± 0.024 | 0.09 | – | 0.00 |
| shuttle | PACE-SHAP | 0.370 ± 0.056 | 0.744 ± 0.044 | 0.381 ± 0.056 | 0.721 ± 0.024 | 1.00 | – | 0.42 |
| shuttle | PACE-SHAP-20 | 0.365 ± 0.053 | 0.747 ± 0.050 | 0.382 ± 0.054 | 0.725 ± 0.024 | 0.59 | – | 0.39 |
| shuttle | PACE-flip | 0.374 ± 0.056 | 0.750 ± 0.045 | 0.374 ± 0.056 | 0.723 ± 0.025 | 1.00 | – | 0.00 |
| shuttle | PACE-multi | 0.375 ± 0.052 | 0.720 ± 0.048 | 0.375 ± 0.052 | 0.736 ± 0.022 | 1.00 | – | 0.04 |
| shuttle | PACE-contrastive | 0.393 ± 0.058 | 0.678 ± 0.046 | 0.392 ± 0.058 | 0.764 ± 0.025 | 1.00 | – | 0.05 |
| shuttle | PACE-decomposition | 0.421 ± 0.059 | 0.708 ± 0.053 | 0.421 ± 0.059 | 0.733 ± 0.025 | 1.00 | – | 0.00 |
| shuttle | KernelSHAP | 0.365 ± 0.053 | 0.747 ± 0.050 | 0.383 ± 0.054 | 0.723 ± 0.023 | 0.59 | – | 0.24 |
| shuttle | KernelSHAP-strong | 0.366 ± 0.053 | 0.747 ± 0.050 | 0.383 ± 0.054 | 0.725 ± 0.024 | – | – | 0.39 |
| shuttle | LIME | 0.381 ± 0.053 | 0.721 ± 0.051 | 0.383 ± 0.054 | 0.732 ± 0.020 | 0.90 | – | 0.39 |
| shuttle | Random | 0.552 ± 0.045 | 0.536 ± 0.061 | 0.552 ± 0.045 | 0.820 ± 0.019 | 0.22 | – | 0.00 |
| magic | PACE-SHAP | -0.052 ± 0.012 | 0.290 ± 0.016 | 0.062 ± 0.021 | 0.500 ± 0.010 | 1.00 | – | 0.44 |
| magic | PACE-SHAP-20 | -0.038 ± 0.014 | 0.279 ± 0.026 | 0.065 ± 0.016 | 0.511 ± 0.017 | 0.50 | – | 0.47 |
| magic | PACE-flip | 0.015 ± 0.010 | 0.234 ± 0.022 | 0.015 ± 0.010 | 0.506 ± 0.015 | 1.00 | – | 0.00 |
| magic | PACE-multi | 0.061 ± 0.015 | 0.170 ± 0.023 | 0.063 ± 0.011 | 0.556 ± 0.011 | 1.00 | – | 0.17 |
| magic | PACE-contrastive | 0.093 ± 0.012 | 0.129 ± 0.025 | 0.094 ± 0.017 | 0.576 ± 0.018 | 1.00 | – | 0.32 |
| magic | PACE-decomposition | 0.094 ± 0.022 | 0.144 ± 0.019 | 0.093 ± 0.022 | 0.588 ± 0.021 | 1.00 | – | 0.00 |
| magic | KernelSHAP | -0.035 ± 0.017 | 0.272 ± 0.027 | 0.069 ± 0.019 | 0.512 ± 0.018 | 0.47 | – | 0.46 |
| magic | KernelSHAP-strong | -0.038 ± 0.014 | 0.279 ± 0.026 | 0.065 ± 0.016 | 0.511 ± 0.017 | – | – | 0.47 |
| magic | LIME | -0.011 ± 0.017 | 0.235 ± 0.027 | 0.069 ± 0.016 | 0.504 ± 0.016 | 0.69 | – | 0.45 |
| magic | Random | 0.140 ± 0.012 | 0.093 ± 0.014 | 0.140 ± 0.012 | 0.641 ± 0.017 | 0.21 | – | 0.00 |
| letter | PACE-SHAP | -0.690 ± 0.016 | -0.000 ± 0.037 | -0.683 ± 0.015 | 0.131 ± 0.009 | 0.99 | – | 0.12 |
| letter | PACE-SHAP-20 | -0.684 ± 0.015 | -0.021 ± 0.035 | -0.675 ± 0.014 | 0.134 ± 0.009 | 0.63 | – | 0.13 |
| letter | PACE-flip | -0.644 ± 0.013 | -0.072 ± 0.035 | -0.644 ± 0.013 | 0.134 ± 0.009 | 1.00 | – | 0.00 |
| letter | PACE-multi | -0.550 ± 0.022 | -0.275 ± 0.032 | -0.538 ± 0.020 | 0.187 ± 0.013 | 1.00 | – | 0.13 |
| letter | PACE-contrastive | -0.457 ± 0.025 | -0.397 ± 0.020 | -0.442 ± 0.017 | 0.248 ± 0.018 | 1.00 | – | 0.23 |
| letter | PACE-decomposition | -0.591 ± 0.021 | -0.160 ± 0.033 | -0.590 ± 0.020 | 0.187 ± 0.012 | 1.00 | – | 0.00 |
| letter | KernelSHAP | -0.678 ± 0.015 | -0.038 ± 0.034 | -0.677 ± 0.015 | 0.136 ± 0.009 | 0.61 | – | 0.00 |
| letter | KernelSHAP-strong | -0.684 ± 0.016 | -0.022 ± 0.035 | -0.675 ± 0.015 | 0.134 ± 0.010 | – | – | 0.13 |
| letter | LIME | -0.656 ± 0.018 | -0.136 ± 0.043 | -0.618 ± 0.017 | 0.139 ± 0.009 | 0.85 | – | 0.35 |
| letter | Random | -0.381 ± 0.022 | -0.423 ± 0.030 | -0.381 ± 0.022 | 0.315 ± 0.015 | 0.13 | – | 0.00 |
| landsat | PACE-SHAP | -0.329 ± 0.025 | 0.231 ± 0.020 | -0.326 ± 0.026 | 0.339 ± 0.015 | 0.94 | – | 0.03 |
| landsat | PACE-SHAP-20 | -0.309 ± 0.028 | 0.206 ± 0.022 | -0.305 ± 0.027 | 0.348 ± 0.016 | 0.52 | – | 0.05 |
| landsat | PACE-flip | -0.255 ± 0.031 | 0.134 ± 0.020 | -0.255 ± 0.031 | 0.376 ± 0.020 | 1.00 | – | 0.00 |
| landsat | PACE-multi | -0.234 ± 0.026 | 0.072 ± 0.021 | -0.222 ± 0.026 | 0.386 ± 0.019 | 1.00 | – | 0.07 |
| landsat | PACE-contrastive | -0.154 ± 0.026 | 0.013 ± 0.021 | -0.141 ± 0.025 | 0.431 ± 0.018 | 1.00 | – | 0.11 |
| landsat | PACE-decomposition | -0.189 ± 0.028 | 0.126 ± 0.030 | -0.189 ± 0.028 | 0.418 ± 0.016 | 1.00 | – | 0.00 |
| landsat | KernelSHAP | -0.217 ± 0.033 | 0.077 ± 0.031 | -0.217 ± 0.033 | 0.406 ± 0.020 | 0.15 | – | 0.00 |
| landsat | KernelSHAP-strong | -0.303 ± 0.027 | 0.203 ± 0.024 | -0.298 ± 0.026 | 0.352 ± 0.016 | – | – | 0.06 |
| landsat | LIME | -0.295 ± 0.029 | 0.126 ± 0.031 | -0.272 ± 0.026 | 0.355 ± 0.018 | 0.52 | – | 0.27 |
| landsat | Random | -0.050 ± 0.021 | -0.057 ± 0.034 | -0.050 ± 0.021 | 0.495 ± 0.013 | 0.06 | – | 0.00 |
| page_blocks | PACE-SHAP | 0.050 ± 0.052 | 0.204 ± 0.074 | 0.087 ± 0.059 | 0.756 ± 0.044 | 1.00 | – | 0.37 |
| page_blocks | PACE-SHAP-20 | 0.042 ± 0.044 | 0.212 ± 0.081 | 0.090 ± 0.058 | 0.739 ± 0.050 | 0.48 | – | 0.47 |
| page_blocks | PACE-flip | 0.051 ± 0.054 | 0.202 ± 0.066 | 0.051 ± 0.054 | 0.692 ± 0.058 | 1.00 | – | 0.00 |
| page_blocks | PACE-multi | 0.056 ± 0.048 | 0.195 ± 0.077 | 0.057 ± 0.050 | 0.706 ± 0.056 | 1.00 | – | 0.11 |
| page_blocks | PACE-contrastive | 0.077 ± 0.040 | 0.131 ± 0.059 | 0.088 ± 0.047 | 0.711 ± 0.048 | 1.00 | – | 0.29 |
| page_blocks | PACE-decomposition | 0.111 ± 0.054 | 0.114 ± 0.057 | 0.110 ± 0.054 | 0.777 ± 0.040 | 1.00 | – | 0.00 |
| page_blocks | KernelSHAP | 0.046 ± 0.043 | 0.205 ± 0.082 | 0.090 ± 0.055 | 0.744 ± 0.042 | 0.47 | – | 0.41 |
| page_blocks | KernelSHAP-strong | 0.043 ± 0.044 | 0.212 ± 0.081 | 0.090 ± 0.058 | 0.743 ± 0.045 | – | – | 0.47 |
| page_blocks | LIME | 0.049 ± 0.043 | 0.196 ± 0.079 | 0.104 ± 0.056 | 0.747 ± 0.053 | 0.79 | – | 0.56 |
| page_blocks | Random | 0.128 ± 0.054 | 0.095 ± 0.061 | 0.128 ± 0.054 | 0.798 ± 0.036 | 0.21 | – | 0.00 |

## PACE-SHAP minus KernelSHAP: deletion margin AUC (lower is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: negative = PACE-SHAP better (more negative).

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.006 ± 0.007 | 2 of 10 | 0.0488 | 0.146 |
| S2 | -0.007 ± 0.006 | 3 of 10 | 0.0273 | 0.137 |
| S3 | -0.014 ± 0.008 | 1 of 10 | 0.00391 | 0.0273 |
| wine | -0.008 ± 0.005 | 0 of 10 | 0.00195 | 0.0215 |
| breast_cancer | -0.063 ± 0.005 | 0 of 10 | 0.00195 | 0.0215 |
| thyroid | -0.026 ± 0.011 | 0 of 10 | 0.00195 | 0.0215 |
| shuttle | +0.005 ± 0.008 | 6 of 10 | 0.275 | 0.551 |
| magic | -0.017 ± 0.014 | 2 of 10 | 0.0273 | 0.137 |
| letter | -0.012 ± 0.007 | 1 of 10 | 0.00586 | 0.0352 |
| landsat | -0.112 ± 0.020 | 0 of 10 | 0.00195 | 0.0215 |
| page_blocks | +0.004 ± 0.018 | 5 of 10 | 0.625 | 0.625 |

## PACE-SHAP-20 minus KernelSHAP: deletion margin AUC (lower is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: negative = PACE-SHAP-20 better (more negative).

| Dataset | Difference | Seeds where PACE-SHAP-20 is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.001 ± 0.003 | 6 of 10 | 0.492 | 1 |
| S2 | -0.004 ± 0.003 | 3 of 10 | 0.0645 | 0.451 |
| S3 | -0.003 ± 0.004 | 2 of 10 | 0.105 | 0.527 |
| wine | -0.002 ± 0.002 | 2 of 10 | 0.0645 | 0.451 |
| breast_cancer | -0.052 ± 0.007 | 0 of 10 | 0.00195 | 0.0215 |
| thyroid | -0.012 ± 0.010 | 0 of 10 | 0.00195 | 0.0215 |
| shuttle | +0.000 ± 0.001 | 4 of 10 | 1 | 1 |
| magic | -0.003 ± 0.005 | 5 of 10 | 0.375 | 1 |
| letter | -0.007 ± 0.002 | 0 of 10 | 0.00195 | 0.0215 |
| landsat | -0.093 ± 0.014 | 0 of 10 | 0.00195 | 0.0215 |
| page_blocks | -0.004 ± 0.005 | 3 of 10 | 0.105 | 0.527 |

## PACE-SHAP minus KernelSHAP-strong: deletion margin AUC (lower is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: negative = PACE-SHAP better (more negative).

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.007 ± 0.008 | 3 of 10 | 0.0371 | 0.26 |
| S2 | -0.003 ± 0.005 | 3 of 10 | 0.131 | 0.422 |
| S3 | -0.011 ± 0.007 | 2 of 10 | 0.00977 | 0.0781 |
| wine | -0.006 ± 0.005 | 3 of 10 | 0.0488 | 0.293 |
| breast_cancer | -0.011 ± 0.005 | 0 of 10 | 0.00195 | 0.0215 |
| thyroid | -0.014 ± 0.007 | 1 of 10 | 0.00391 | 0.0352 |
| shuttle | +0.004 ± 0.007 | 7 of 10 | 0.232 | 0.465 |
| magic | -0.015 ± 0.014 | 4 of 10 | 0.105 | 0.422 |
| letter | -0.006 ± 0.005 | 2 of 10 | 0.0488 | 0.293 |
| landsat | -0.026 ± 0.014 | 0 of 10 | 0.00195 | 0.0215 |
| page_blocks | +0.007 ± 0.017 | 6 of 10 | 0.625 | 0.625 |

## PACE-SHAP minus LIME: deletion margin AUC (lower is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: negative = PACE-SHAP better (more negative).

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.029 ± 0.005 | 0 of 10 | 0.00195 | 0.0215 |
| S2 | -0.037 ± 0.010 | 0 of 10 | 0.00195 | 0.0215 |
| S3 | -0.067 ± 0.012 | 0 of 10 | 0.00195 | 0.0215 |
| wine | -0.037 ± 0.006 | 0 of 10 | 0.00195 | 0.0215 |
| breast_cancer | -0.030 ± 0.008 | 0 of 10 | 0.00195 | 0.0215 |
| thyroid | -0.022 ± 0.013 | 2 of 10 | 0.00977 | 0.0293 |
| shuttle | -0.011 ± 0.014 | 3 of 10 | 0.16 | 0.32 |
| magic | -0.041 ± 0.011 | 0 of 10 | 0.00195 | 0.0215 |
| letter | -0.034 ± 0.009 | 0 of 10 | 0.00195 | 0.0215 |
| landsat | -0.033 ± 0.013 | 0 of 10 | 0.00195 | 0.0215 |
| page_blocks | +0.001 ± 0.022 | 4 of 10 | 0.846 | 0.846 |

## PACE-SHAP minus PACE-flip: deletion margin AUC (lower is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: negative = PACE-SHAP better (more negative).

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.050 ± 0.008 | 0 of 10 | 0.00195 | 0.0215 |
| S2 | -0.056 ± 0.006 | 0 of 10 | 0.00195 | 0.0215 |
| S3 | -0.061 ± 0.010 | 0 of 10 | 0.00195 | 0.0215 |
| wine | -0.022 ± 0.007 | 0 of 10 | 0.00195 | 0.0215 |
| breast_cancer | -0.025 ± 0.006 | 0 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.005 ± 0.011 | 8 of 10 | 0.131 | 0.393 |
| shuttle | -0.004 ± 0.012 | 3 of 10 | 0.16 | 0.393 |
| magic | -0.067 ± 0.012 | 0 of 10 | 0.00195 | 0.0215 |
| letter | -0.046 ± 0.010 | 0 of 10 | 0.00195 | 0.0215 |
| landsat | -0.074 ± 0.019 | 0 of 10 | 0.00195 | 0.0215 |
| page_blocks | -0.001 ± 0.010 | 3 of 10 | 0.695 | 0.695 |

## PACE-SHAP minus KernelSHAP: insertion margin AUC (higher is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: positive = PACE-SHAP better.

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.010 ± 0.006 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.018 ± 0.012 | 10 of 10 | 0.00195 | 0.0215 |
| S3 | +0.013 ± 0.008 | 8 of 10 | 0.0195 | 0.0781 |
| wine | +0.009 ± 0.006 | 9 of 10 | 0.00586 | 0.0293 |
| breast_cancer | +0.066 ± 0.010 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.023 ± 0.010 | 10 of 10 | 0.00195 | 0.0215 |
| shuttle | -0.003 ± 0.010 | 5 of 10 | 0.492 | 0.984 |
| magic | +0.018 ± 0.013 | 8 of 10 | 0.0273 | 0.082 |
| letter | +0.037 ± 0.008 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.154 ± 0.022 | 10 of 10 | 0.00195 | 0.0215 |
| page_blocks | -0.001 ± 0.021 | 6 of 10 | 0.625 | 0.984 |

## PACE-SHAP-20 minus KernelSHAP: insertion margin AUC (higher is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: positive = PACE-SHAP-20 better.

| Dataset | Difference | Seeds where PACE-SHAP-20 is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.001 ± 0.003 | 6 of 10 | 0.432 | 1 |
| S2 | +0.001 ± 0.004 | 7 of 10 | 0.492 | 1 |
| S3 | +0.000 ± 0.004 | 5 of 10 | 0.922 | 1 |
| wine | +0.002 ± 0.002 | 7 of 10 | 0.0645 | 0.322 |
| breast_cancer | +0.054 ± 0.009 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.010 ± 0.007 | 10 of 10 | 0.00195 | 0.0215 |
| shuttle | +0.000 ± 0.001 | 6 of 10 | 0.625 | 1 |
| magic | +0.007 ± 0.004 | 9 of 10 | 0.00391 | 0.0273 |
| letter | +0.017 ± 0.004 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.129 ± 0.021 | 10 of 10 | 0.00195 | 0.0215 |
| page_blocks | +0.007 ± 0.006 | 9 of 10 | 0.00391 | 0.0273 |

## PACE-SHAP minus KernelSHAP-strong: insertion margin AUC (higher is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: positive = PACE-SHAP better.

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.009 ± 0.005 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.016 ± 0.012 | 8 of 10 | 0.0137 | 0.0684 |
| S3 | +0.013 ± 0.006 | 10 of 10 | 0.00195 | 0.0215 |
| wine | +0.006 ± 0.006 | 8 of 10 | 0.00977 | 0.0684 |
| breast_cancer | +0.012 ± 0.005 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.012 ± 0.009 | 8 of 10 | 0.0273 | 0.109 |
| shuttle | -0.003 ± 0.009 | 4 of 10 | 0.492 | 0.984 |
| magic | +0.011 ± 0.013 | 6 of 10 | 0.16 | 0.48 |
| letter | +0.021 ± 0.008 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.027 ± 0.021 | 9 of 10 | 0.00977 | 0.0684 |
| page_blocks | -0.008 ± 0.023 | 3 of 10 | 0.625 | 0.984 |

## PACE-SHAP minus LIME: insertion margin AUC (higher is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: positive = PACE-SHAP better.

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.036 ± 0.005 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.045 ± 0.011 | 10 of 10 | 0.00195 | 0.0215 |
| S3 | +0.070 ± 0.009 | 10 of 10 | 0.00195 | 0.0215 |
| wine | +0.041 ± 0.006 | 10 of 10 | 0.00195 | 0.0215 |
| breast_cancer | +0.019 ± 0.006 | 10 of 10 | 0.00195 | 0.0215 |
| thyroid | +0.041 ± 0.010 | 10 of 10 | 0.00195 | 0.0215 |
| shuttle | +0.023 ± 0.016 | 9 of 10 | 0.0137 | 0.0273 |
| magic | +0.056 ± 0.026 | 10 of 10 | 0.00195 | 0.0215 |
| letter | +0.135 ± 0.017 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.105 ± 0.017 | 10 of 10 | 0.00195 | 0.0215 |
| page_blocks | +0.007 ± 0.023 | 7 of 10 | 0.232 | 0.232 |

## PACE-SHAP minus PACE-flip: insertion margin AUC (higher is better)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Sign: positive = PACE-SHAP better.

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.029 ± 0.006 | 10 of 10 | 0.00195 | 0.0215 |
| S2 | +0.048 ± 0.010 | 10 of 10 | 0.00195 | 0.0215 |
| S3 | +0.044 ± 0.009 | 10 of 10 | 0.00195 | 0.0215 |
| wine | +0.044 ± 0.009 | 10 of 10 | 0.00195 | 0.0215 |
| breast_cancer | +0.014 ± 0.009 | 8 of 10 | 0.00977 | 0.0391 |
| thyroid | -0.003 ± 0.009 | 4 of 10 | 0.375 | 0.75 |
| shuttle | -0.006 ± 0.010 | 3 of 10 | 0.16 | 0.48 |
| magic | +0.056 ± 0.009 | 10 of 10 | 0.00195 | 0.0215 |
| letter | +0.072 ± 0.011 | 10 of 10 | 0.00195 | 0.0215 |
| landsat | +0.097 ± 0.009 | 10 of 10 | 0.00195 | 0.0215 |
| page_blocks | +0.002 ± 0.025 | 4 of 10 | 1 | 1 |

## The model's own decision sensitivity (synthetic sets)

Share of all single-feature decision flips (PACE-flip's flip mass, computed from the model) that come from the truly relevant features 0–2, and share of points where replacing some irrelevant feature can flip the decision.

| Dataset | Flip mass on relevant features | Points with an irrelevant feature that flips |
|---|---|---|
| S1 | 0.471 ± 0.030 | 0.696 ± 0.041 |
| S2 | 0.417 ± 0.022 | 0.826 ± 0.026 |
| S3 | 0.464 ± 0.044 | 0.760 ± 0.023 |
