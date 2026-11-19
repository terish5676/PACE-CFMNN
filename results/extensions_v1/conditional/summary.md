# RQ1 extension B3: interventional vs conditional replacement

50 test points per dataset and seed; seeds [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]. Every explainer uses the same 200 explainer rows (half the training set if smaller); the tests copy values from a disjoint set of evaluation rows. Interventional: replaced features come from a random row. Conditional: they come from rows close to the point on the features it keeps (PACE-SHAP-cond and KernelSHAP-cond: Gaussian weights, sigma = 0.1, on standardised distances; PACE-flip-cond: the 32 nearest rows on the other features; conditional tests: the q-th nearest evaluation row in repetition q). KernelSHAP and KernelSHAP-cond share one implementation (200 coalitions per point) and differ only in the row weights. Each such pair (the two KernelSHAPs; the two PACE-SHAPs where exact, d ≤ 12) makes its model calls once; a paired method's time is the shared part plus its own part. The two kinds of test answer different questions, so they are reported separately and never combined. Mean ± 95% CI over seeds.


## Interventional test

Deletion margin AUC: lower = more faithful. Insertion margin AUC: higher = more faithful.

| Dataset | Method | Deletion margin AUC | Insertion margin AUC |
|---|---|---|---|
| S1 | PACE-SHAP | -0.126 ± 0.045 | 0.662 ± 0.009 |
| S1 | PACE-SHAP-cond | -0.088 ± 0.047 | 0.627 ± 0.013 |
| S1 | PACE-flip | -0.074 ± 0.048 | 0.631 ± 0.017 |
| S1 | PACE-flip-cond | -0.070 ± 0.045 | 0.625 ± 0.017 |
| S1 | KernelSHAP | -0.125 ± 0.045 | 0.661 ± 0.010 |
| S1 | KernelSHAP-cond | -0.080 ± 0.047 | 0.618 ± 0.012 |
| S1 | LIME | -0.094 ± 0.046 | 0.624 ± 0.012 |
| S1 | Random | 0.279 ± 0.034 | 0.301 ± 0.040 |
| S2 | PACE-SHAP | -0.142 ± 0.032 | 0.616 ± 0.020 |
| S2 | PACE-SHAP-cond | -0.089 ± 0.033 | 0.569 ± 0.017 |
| S2 | PACE-flip | -0.089 ± 0.033 | 0.568 ± 0.018 |
| S2 | PACE-flip-cond | -0.084 ± 0.030 | 0.564 ± 0.017 |
| S2 | KernelSHAP | -0.140 ± 0.033 | 0.613 ± 0.019 |
| S2 | KernelSHAP-cond | -0.078 ± 0.031 | 0.556 ± 0.018 |
| S2 | LIME | -0.107 ± 0.033 | 0.568 ± 0.017 |
| S2 | Random | 0.247 ± 0.026 | 0.273 ± 0.037 |
| S3 | PACE-SHAP | -0.053 ± 0.025 | 0.651 ± 0.023 |
| S3 | PACE-SHAP-cond | -0.006 ± 0.026 | 0.615 ± 0.023 |
| S3 | PACE-flip | 0.007 ± 0.028 | 0.612 ± 0.023 |
| S3 | PACE-flip-cond | 0.013 ± 0.028 | 0.609 ± 0.023 |
| S3 | KernelSHAP | -0.046 ± 0.024 | 0.649 ± 0.023 |
| S3 | KernelSHAP-cond | 0.006 ± 0.026 | 0.598 ± 0.023 |
| S3 | LIME | 0.015 ± 0.015 | 0.571 ± 0.029 |
| S3 | Random | 0.327 ± 0.028 | 0.315 ± 0.023 |
| S1c | PACE-SHAP | -0.063 ± 0.044 | 0.702 ± 0.012 |
| S1c | PACE-SHAP-cond | -0.030 ± 0.045 | 0.675 ± 0.013 |
| S1c | PACE-flip | -0.016 ± 0.043 | 0.677 ± 0.013 |
| S1c | PACE-flip-cond | 0.004 ± 0.041 | 0.655 ± 0.015 |
| S1c | KernelSHAP | -0.062 ± 0.045 | 0.702 ± 0.012 |
| S1c | KernelSHAP-cond | -0.024 ± 0.045 | 0.667 ± 0.014 |
| S1c | LIME | -0.048 ± 0.043 | 0.678 ± 0.013 |
| S1c | Random | 0.342 ± 0.028 | 0.335 ± 0.021 |
| S2c | PACE-SHAP | -0.129 ± 0.027 | 0.661 ± 0.016 |
| S2c | PACE-SHAP-cond | -0.092 ± 0.030 | 0.627 ± 0.018 |
| S2c | PACE-flip | -0.079 ± 0.027 | 0.612 ± 0.017 |
| S2c | PACE-flip-cond | -0.054 ± 0.025 | 0.585 ± 0.017 |
| S2c | KernelSHAP | -0.127 ± 0.028 | 0.659 ± 0.018 |
| S2c | KernelSHAP-cond | -0.078 ± 0.029 | 0.619 ± 0.020 |
| S2c | LIME | -0.103 ± 0.030 | 0.630 ± 0.018 |
| S2c | Random | 0.278 ± 0.028 | 0.276 ± 0.029 |
| wine | PACE-SHAP | -0.072 ± 0.024 | 0.542 ± 0.023 |
| wine | PACE-SHAP-cond | -0.000 ± 0.028 | 0.454 ± 0.026 |
| wine | PACE-flip | -0.053 ± 0.025 | 0.502 ± 0.027 |
| wine | PACE-flip-cond | -0.033 ± 0.027 | 0.468 ± 0.028 |
| wine | KernelSHAP | -0.073 ± 0.024 | 0.541 ± 0.023 |
| wine | KernelSHAP-cond | 0.000 ± 0.026 | 0.447 ± 0.029 |
| wine | LIME | -0.034 ± 0.027 | 0.495 ± 0.029 |
| wine | Random | 0.243 ± 0.020 | 0.236 ± 0.030 |
| breast_cancer | PACE-SHAP | 0.135 ± 0.035 | 0.709 ± 0.016 |
| breast_cancer | PACE-SHAP-cond | 0.222 ± 0.031 | 0.601 ± 0.014 |
| breast_cancer | PACE-flip | 0.159 ± 0.032 | 0.699 ± 0.022 |
| breast_cancer | PACE-flip-cond | 0.212 ± 0.036 | 0.586 ± 0.024 |
| breast_cancer | KernelSHAP | 0.137 ± 0.034 | 0.706 ± 0.016 |
| breast_cancer | KernelSHAP-cond | 0.271 ± 0.029 | 0.559 ± 0.018 |
| breast_cancer | LIME | 0.164 ± 0.031 | 0.684 ± 0.023 |
| breast_cancer | Random | 0.418 ± 0.027 | 0.423 ± 0.020 |
| thyroid | PACE-SHAP | 0.508 ± 0.042 | 0.893 ± 0.026 |
| thyroid | PACE-SHAP-cond | 0.543 ± 0.042 | 0.853 ± 0.029 |
| thyroid | PACE-flip | 0.503 ± 0.043 | 0.900 ± 0.024 |
| thyroid | PACE-flip-cond | 0.513 ± 0.041 | 0.889 ± 0.027 |
| thyroid | KernelSHAP | 0.508 ± 0.042 | 0.892 ± 0.026 |
| thyroid | KernelSHAP-cond | 0.557 ± 0.043 | 0.844 ± 0.031 |
| thyroid | LIME | 0.543 ± 0.050 | 0.853 ± 0.025 |
| thyroid | Random | 0.709 ± 0.038 | 0.708 ± 0.030 |
| shuttle | PACE-SHAP | 0.310 ± 0.065 | 0.706 ± 0.044 |
| shuttle | PACE-SHAP-cond | 0.341 ± 0.068 | 0.671 ± 0.045 |
| shuttle | PACE-flip | 0.325 ± 0.055 | 0.703 ± 0.038 |
| shuttle | PACE-flip-cond | 0.322 ± 0.059 | 0.685 ± 0.050 |
| shuttle | KernelSHAP | 0.311 ± 0.065 | 0.706 ± 0.044 |
| shuttle | KernelSHAP-cond | 0.355 ± 0.068 | 0.650 ± 0.046 |
| shuttle | LIME | 0.328 ± 0.067 | 0.675 ± 0.046 |
| shuttle | Random | 0.497 ± 0.054 | 0.489 ± 0.059 |
| magic | PACE-SHAP | -0.047 ± 0.016 | 0.268 ± 0.026 |
| magic | PACE-SHAP-cond | 0.055 ± 0.016 | 0.158 ± 0.024 |
| magic | PACE-flip | 0.023 ± 0.012 | 0.208 ± 0.016 |
| magic | PACE-flip-cond | 0.046 ± 0.017 | 0.192 ± 0.016 |
| magic | KernelSHAP | -0.050 ± 0.017 | 0.267 ± 0.025 |
| magic | KernelSHAP-cond | 0.070 ± 0.015 | 0.146 ± 0.025 |
| magic | LIME | -0.007 ± 0.018 | 0.216 ± 0.033 |
| magic | Random | 0.129 ± 0.019 | 0.082 ± 0.019 |
| letter | PACE-SHAP | -0.703 ± 0.018 | -0.015 ± 0.053 |
| letter | PACE-SHAP-cond | -0.607 ± 0.020 | -0.210 ± 0.055 |
| letter | PACE-flip | -0.653 ± 0.020 | -0.085 ± 0.058 |
| letter | PACE-flip-cond | -0.632 ± 0.020 | -0.114 ± 0.053 |
| letter | KernelSHAP | -0.702 ± 0.018 | -0.016 ± 0.052 |
| letter | KernelSHAP-cond | -0.606 ± 0.019 | -0.219 ± 0.056 |
| letter | LIME | -0.666 ± 0.021 | -0.130 ± 0.045 |
| letter | Random | -0.408 ± 0.035 | -0.413 ± 0.032 |
| landsat | PACE-SHAP | -0.330 ± 0.036 | 0.249 ± 0.038 |
| landsat | PACE-SHAP-cond | -0.153 ± 0.033 | 0.047 ± 0.046 |
| landsat | PACE-flip | -0.237 ± 0.038 | 0.137 ± 0.027 |
| landsat | PACE-flip-cond | -0.183 ± 0.039 | 0.048 ± 0.043 |
| landsat | KernelSHAP | -0.285 ± 0.034 | 0.213 ± 0.039 |
| landsat | KernelSHAP-cond | -0.090 ± 0.029 | -0.003 ± 0.039 |
| landsat | LIME | -0.286 ± 0.040 | 0.143 ± 0.041 |
| landsat | Random | -0.045 ± 0.035 | -0.049 ± 0.031 |
| page_blocks | PACE-SHAP | 0.041 ± 0.045 | 0.276 ± 0.077 |
| page_blocks | PACE-SHAP-cond | 0.077 ± 0.051 | 0.226 ± 0.079 |
| page_blocks | PACE-flip | 0.056 ± 0.053 | 0.262 ± 0.071 |
| page_blocks | PACE-flip-cond | 0.051 ± 0.055 | 0.265 ± 0.083 |
| page_blocks | KernelSHAP | 0.041 ± 0.045 | 0.273 ± 0.076 |
| page_blocks | KernelSHAP-cond | 0.088 ± 0.056 | 0.216 ± 0.083 |
| page_blocks | LIME | 0.057 ± 0.055 | 0.244 ± 0.073 |
| page_blocks | Random | 0.160 ± 0.059 | 0.132 ± 0.065 |

## Conditional test

Deletion margin AUC: lower = more faithful. Insertion margin AUC: higher = more faithful.

| Dataset | Method | Deletion margin AUC | Insertion margin AUC |
|---|---|---|---|
| S1 | PACE-SHAP | -0.184 ± 0.041 | 0.778 ± 0.019 |
| S1 | PACE-SHAP-cond | -0.131 ± 0.039 | 0.738 ± 0.023 |
| S1 | PACE-flip | -0.107 ± 0.037 | 0.733 ± 0.024 |
| S1 | PACE-flip-cond | -0.102 ± 0.038 | 0.728 ± 0.025 |
| S1 | KernelSHAP | -0.184 ± 0.039 | 0.777 ± 0.020 |
| S1 | KernelSHAP-cond | -0.120 ± 0.042 | 0.729 ± 0.023 |
| S1 | LIME | -0.155 ± 0.036 | 0.744 ± 0.024 |
| S1 | Random | 0.309 ± 0.039 | 0.329 ± 0.053 |
| S2 | PACE-SHAP | -0.172 ± 0.030 | 0.713 ± 0.019 |
| S2 | PACE-SHAP-cond | -0.110 ± 0.029 | 0.660 ± 0.022 |
| S2 | PACE-flip | -0.094 ± 0.024 | 0.652 ± 0.023 |
| S2 | PACE-flip-cond | -0.089 ± 0.029 | 0.648 ± 0.022 |
| S2 | KernelSHAP | -0.163 ± 0.033 | 0.713 ± 0.019 |
| S2 | KernelSHAP-cond | -0.092 ± 0.030 | 0.649 ± 0.022 |
| S2 | LIME | -0.143 ± 0.026 | 0.672 ± 0.017 |
| S2 | Random | 0.276 ± 0.026 | 0.296 ± 0.032 |
| S3 | PACE-SHAP | -0.089 ± 0.010 | 0.752 ± 0.023 |
| S3 | PACE-SHAP-cond | -0.039 ± 0.012 | 0.711 ± 0.023 |
| S3 | PACE-flip | -0.009 ± 0.022 | 0.699 ± 0.018 |
| S3 | PACE-flip-cond | -0.003 ± 0.022 | 0.694 ± 0.017 |
| S3 | KernelSHAP | -0.087 ± 0.011 | 0.750 ± 0.024 |
| S3 | KernelSHAP-cond | -0.023 ± 0.011 | 0.692 ± 0.025 |
| S3 | LIME | -0.012 ± 0.014 | 0.674 ± 0.030 |
| S3 | Random | 0.351 ± 0.022 | 0.344 ± 0.020 |
| S1c | PACE-SHAP | -0.094 ± 0.030 | 0.812 ± 0.010 |
| S1c | PACE-SHAP-cond | -0.052 ± 0.035 | 0.787 ± 0.013 |
| S1c | PACE-flip | -0.040 ± 0.029 | 0.785 ± 0.014 |
| S1c | PACE-flip-cond | -0.012 ± 0.029 | 0.766 ± 0.013 |
| S1c | KernelSHAP | -0.092 ± 0.031 | 0.809 ± 0.010 |
| S1c | KernelSHAP-cond | -0.042 ± 0.038 | 0.780 ± 0.017 |
| S1c | LIME | -0.082 ± 0.032 | 0.787 ± 0.013 |
| S1c | Random | 0.427 ± 0.036 | 0.433 ± 0.028 |
| S2c | PACE-SHAP | -0.157 ± 0.030 | 0.743 ± 0.014 |
| S2c | PACE-SHAP-cond | -0.105 ± 0.029 | 0.706 ± 0.015 |
| S2c | PACE-flip | -0.082 ± 0.027 | 0.680 ± 0.015 |
| S2c | PACE-flip-cond | -0.054 ± 0.029 | 0.658 ± 0.016 |
| S2c | KernelSHAP | -0.153 ± 0.030 | 0.739 ± 0.013 |
| S2c | KernelSHAP-cond | -0.091 ± 0.030 | 0.701 ± 0.015 |
| S2c | LIME | -0.133 ± 0.030 | 0.717 ± 0.016 |
| S2c | Random | 0.347 ± 0.040 | 0.333 ± 0.026 |
| wine | PACE-SHAP | 0.080 ± 0.045 | 0.891 ± 0.006 |
| wine | PACE-SHAP-cond | 0.148 ± 0.047 | 0.857 ± 0.010 |
| wine | PACE-flip | 0.116 ± 0.047 | 0.866 ± 0.011 |
| wine | PACE-flip-cond | 0.115 ± 0.044 | 0.849 ± 0.015 |
| wine | KernelSHAP | 0.078 ± 0.044 | 0.890 ± 0.007 |
| wine | KernelSHAP-cond | 0.163 ± 0.045 | 0.861 ± 0.007 |
| wine | LIME | 0.133 ± 0.042 | 0.867 ± 0.009 |
| wine | Random | 0.594 ± 0.034 | 0.582 ± 0.034 |
| breast_cancer | PACE-SHAP | 0.249 ± 0.027 | 0.933 ± 0.006 |
| breast_cancer | PACE-SHAP-cond | 0.360 ± 0.033 | 0.915 ± 0.010 |
| breast_cancer | PACE-flip | 0.331 ± 0.029 | 0.914 ± 0.012 |
| breast_cancer | PACE-flip-cond | 0.398 ± 0.035 | 0.897 ± 0.021 |
| breast_cancer | KernelSHAP | 0.257 ± 0.028 | 0.932 ± 0.006 |
| breast_cancer | KernelSHAP-cond | 0.471 ± 0.032 | 0.898 ± 0.009 |
| breast_cancer | LIME | 0.244 ± 0.021 | 0.920 ± 0.009 |
| breast_cancer | Random | 0.736 ± 0.021 | 0.733 ± 0.020 |
| thyroid | PACE-SHAP | 0.528 ± 0.054 | 0.907 ± 0.024 |
| thyroid | PACE-SHAP-cond | 0.556 ± 0.050 | 0.875 ± 0.028 |
| thyroid | PACE-flip | 0.527 ± 0.052 | 0.917 ± 0.022 |
| thyroid | PACE-flip-cond | 0.536 ± 0.051 | 0.909 ± 0.024 |
| thyroid | KernelSHAP | 0.528 ± 0.054 | 0.907 ± 0.024 |
| thyroid | KernelSHAP-cond | 0.569 ± 0.051 | 0.868 ± 0.030 |
| thyroid | LIME | 0.551 ± 0.062 | 0.872 ± 0.024 |
| thyroid | Random | 0.735 ± 0.039 | 0.729 ± 0.031 |
| shuttle | PACE-SHAP | 0.482 ± 0.057 | 0.773 ± 0.035 |
| shuttle | PACE-SHAP-cond | 0.452 ± 0.061 | 0.771 ± 0.038 |
| shuttle | PACE-flip | 0.505 ± 0.049 | 0.771 ± 0.037 |
| shuttle | PACE-flip-cond | 0.490 ± 0.052 | 0.786 ± 0.039 |
| shuttle | KernelSHAP | 0.481 ± 0.057 | 0.772 ± 0.035 |
| shuttle | KernelSHAP-cond | 0.469 ± 0.060 | 0.758 ± 0.039 |
| shuttle | LIME | 0.511 ± 0.057 | 0.761 ± 0.041 |
| shuttle | Random | 0.639 ± 0.047 | 0.628 ± 0.043 |
| magic | PACE-SHAP | 0.050 ± 0.019 | 0.187 ± 0.029 |
| magic | PACE-SHAP-cond | 0.047 ± 0.014 | 0.174 ± 0.031 |
| magic | PACE-flip | 0.064 ± 0.015 | 0.177 ± 0.023 |
| magic | PACE-flip-cond | 0.057 ± 0.010 | 0.203 ± 0.028 |
| magic | KernelSHAP | 0.050 ± 0.017 | 0.186 ± 0.027 |
| magic | KernelSHAP-cond | 0.060 ± 0.017 | 0.164 ± 0.029 |
| magic | LIME | 0.071 ± 0.014 | 0.157 ± 0.031 |
| magic | Random | 0.140 ± 0.018 | 0.097 ± 0.014 |
| letter | PACE-SHAP | -0.600 ± 0.020 | 0.079 ± 0.064 |
| letter | PACE-SHAP-cond | -0.525 ± 0.024 | -0.057 ± 0.062 |
| letter | PACE-flip | -0.520 ± 0.023 | 0.023 ± 0.070 |
| letter | PACE-flip-cond | -0.521 ± 0.023 | 0.024 ± 0.069 |
| letter | KernelSHAP | -0.599 ± 0.022 | 0.078 ± 0.062 |
| letter | KernelSHAP-cond | -0.517 ± 0.024 | -0.060 ± 0.063 |
| letter | LIME | -0.547 ± 0.029 | -0.042 ± 0.052 |
| letter | Random | -0.272 ± 0.038 | -0.275 ± 0.044 |
| landsat | PACE-SHAP | 0.160 ± 0.043 | 0.597 ± 0.040 |
| landsat | PACE-SHAP-cond | 0.265 ± 0.029 | 0.598 ± 0.048 |
| landsat | PACE-flip | 0.277 ± 0.048 | 0.590 ± 0.040 |
| landsat | PACE-flip-cond | 0.193 ± 0.043 | 0.639 ± 0.051 |
| landsat | KernelSHAP | 0.229 ± 0.045 | 0.600 ± 0.039 |
| landsat | KernelSHAP-cond | 0.411 ± 0.035 | 0.550 ± 0.055 |
| landsat | LIME | 0.165 ± 0.040 | 0.536 ± 0.046 |
| landsat | Random | 0.483 ± 0.048 | 0.476 ± 0.042 |
| page_blocks | PACE-SHAP | 0.110 ± 0.050 | 0.253 ± 0.063 |
| page_blocks | PACE-SHAP-cond | 0.114 ± 0.049 | 0.253 ± 0.070 |
| page_blocks | PACE-flip | 0.119 ± 0.044 | 0.250 ± 0.074 |
| page_blocks | PACE-flip-cond | 0.098 ± 0.039 | 0.273 ± 0.078 |
| page_blocks | KernelSHAP | 0.110 ± 0.050 | 0.249 ± 0.062 |
| page_blocks | KernelSHAP-cond | 0.117 ± 0.050 | 0.244 ± 0.072 |
| page_blocks | LIME | 0.141 ± 0.068 | 0.217 ± 0.051 |
| page_blocks | Random | 0.184 ± 0.047 | 0.160 ± 0.061 |

## PACE-SHAP-cond minus KernelSHAP-cond: conditional deletion margin AUC (primary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for PACE-SHAP-cond: negative.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.011 ± 0.011 | 4 of 10 | 0.131 | 0.393 |
| S2 | -0.017 ± 0.010 | 2 of 10 | 0.00977 | 0.0684 |
| S3 | -0.016 ± 0.009 | 0 of 10 | 0.00195 | 0.0254 |
| S1c | -0.010 ± 0.009 | 2 of 10 | 0.0195 | 0.0977 |
| S2c | -0.014 ± 0.008 | 2 of 10 | 0.00977 | 0.0684 |
| wine | -0.015 ± 0.019 | 3 of 10 | 0.16 | 0.393 |
| breast_cancer | -0.111 ± 0.012 | 0 of 10 | 0.00195 | 0.0254 |
| thyroid | -0.013 ± 0.007 | 0 of 10 | 0.00195 | 0.0254 |
| shuttle | -0.017 ± 0.008 | 0 of 10 | 0.00195 | 0.0254 |
| magic | -0.013 ± 0.008 | 0 of 10 | 0.00195 | 0.0254 |
| letter | -0.008 ± 0.007 | 2 of 10 | 0.0371 | 0.148 |
| landsat | -0.146 ± 0.014 | 0 of 10 | 0.00195 | 0.0254 |
| page_blocks | -0.003 ± 0.007 | 3 of 10 | 0.322 | 0.393 |

## PACE-SHAP-cond minus KernelSHAP-cond: conditional insertion margin AUC (primary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for PACE-SHAP-cond: positive.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.009 ± 0.004 | 9 of 10 | 0.00391 | 0.0352 |
| S2 | +0.011 ± 0.009 | 8 of 10 | 0.0273 | 0.137 |
| S3 | +0.019 ± 0.007 | 10 of 10 | 0.00195 | 0.0254 |
| S1c | +0.008 ± 0.006 | 6 of 10 | 0.105 | 0.316 |
| S2c | +0.005 ± 0.004 | 9 of 10 | 0.0195 | 0.117 |
| wine | -0.004 ± 0.005 | 3 of 10 | 0.105 | 0.316 |
| breast_cancer | +0.017 ± 0.006 | 10 of 10 | 0.00195 | 0.0254 |
| thyroid | +0.007 ± 0.007 | 7 of 10 | 0.0273 | 0.137 |
| shuttle | +0.012 ± 0.007 | 9 of 10 | 0.00391 | 0.0352 |
| magic | +0.011 ± 0.008 | 8 of 10 | 0.0137 | 0.0957 |
| letter | +0.003 ± 0.010 | 7 of 10 | 0.432 | 0.432 |
| landsat | +0.048 ± 0.012 | 10 of 10 | 0.00195 | 0.0254 |
| page_blocks | +0.009 ± 0.007 | 10 of 10 | 0.00195 | 0.0254 |

## PACE-SHAP minus KernelSHAP: interventional deletion margin AUC (secondary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for PACE-SHAP: negative.

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.001 ± 0.002 | 5 of 10 | 0.375 | 1 |
| S2 | -0.002 ± 0.003 | 2 of 10 | 0.084 | 0.756 |
| S3 | -0.007 ± 0.003 | 0 of 10 | 0.00195 | 0.0254 |
| S1c | -0.001 ± 0.004 | 6 of 10 | 0.922 | 1 |
| S2c | -0.002 ± 0.003 | 3 of 10 | 0.105 | 0.756 |
| wine | +0.001 ± 0.001 | 7 of 10 | 0.193 | 1 |
| breast_cancer | -0.002 ± 0.001 | 0 of 10 | 0.00195 | 0.0254 |
| thyroid | +0.000 ± 0.001 | 4 of 10 | 0.846 | 1 |
| shuttle | -0.001 ± 0.001 | 2 of 10 | 0.193 | 1 |
| magic | +0.002 ± 0.003 | 8 of 10 | 0.0488 | 0.488 |
| letter | -0.001 ± 0.001 | 1 of 10 | 0.084 | 0.756 |
| landsat | -0.045 ± 0.009 | 0 of 10 | 0.00195 | 0.0254 |
| page_blocks | -0.000 ± 0.003 | 5 of 10 | 0.846 | 1 |

## PACE-SHAP minus KernelSHAP: interventional insertion margin AUC (secondary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for PACE-SHAP: positive.

| Dataset | Difference | Seeds where PACE-SHAP is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.001 ± 0.002 | 5 of 10 | 0.695 | 1 |
| S2 | +0.004 ± 0.003 | 8 of 10 | 0.084 | 0.84 |
| S3 | +0.003 ± 0.003 | 8 of 10 | 0.0488 | 0.537 |
| S1c | +0.001 ± 0.002 | 6 of 10 | 0.557 | 1 |
| S2c | +0.002 ± 0.004 | 8 of 10 | 0.232 | 1 |
| wine | +0.000 ± 0.002 | 6 of 10 | 0.846 | 1 |
| breast_cancer | +0.003 ± 0.001 | 10 of 10 | 0.00195 | 0.0254 |
| thyroid | +0.001 ± 0.001 | 5 of 10 | 0.846 | 1 |
| shuttle | +0.000 ± 0.001 | 6 of 10 | 0.625 | 1 |
| magic | +0.001 ± 0.005 | 7 of 10 | 0.625 | 1 |
| letter | +0.001 ± 0.004 | 6 of 10 | 0.492 | 1 |
| landsat | +0.036 ± 0.007 | 10 of 10 | 0.00195 | 0.0254 |
| page_blocks | +0.002 ± 0.004 | 7 of 10 | 0.16 | 1 |

## PACE-SHAP-cond minus PACE-SHAP: conditional deletion margin AUC (secondary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for PACE-SHAP-cond: negative.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.053 ± 0.012 | 10 of 10 | 0.00195 | 0.0254 |
| S2 | +0.063 ± 0.014 | 10 of 10 | 0.00195 | 0.0254 |
| S3 | +0.050 ± 0.008 | 10 of 10 | 0.00195 | 0.0254 |
| S1c | +0.043 ± 0.011 | 10 of 10 | 0.00195 | 0.0254 |
| S2c | +0.052 ± 0.008 | 10 of 10 | 0.00195 | 0.0254 |
| wine | +0.068 ± 0.015 | 10 of 10 | 0.00195 | 0.0254 |
| breast_cancer | +0.112 ± 0.016 | 10 of 10 | 0.00195 | 0.0254 |
| thyroid | +0.029 ± 0.016 | 8 of 10 | 0.00977 | 0.0293 |
| shuttle | -0.029 ± 0.013 | 1 of 10 | 0.00391 | 0.0254 |
| magic | -0.003 ± 0.012 | 4 of 10 | 0.432 | 0.863 |
| letter | +0.076 ± 0.012 | 10 of 10 | 0.00195 | 0.0254 |
| landsat | +0.106 ± 0.026 | 10 of 10 | 0.00195 | 0.0254 |
| page_blocks | +0.004 ± 0.014 | 5 of 10 | 0.695 | 0.863 |

## PACE-SHAP-cond minus PACE-SHAP: conditional insertion margin AUC (secondary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for PACE-SHAP-cond: positive.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.040 ± 0.006 | 0 of 10 | 0.00195 | 0.0254 |
| S2 | -0.054 ± 0.011 | 0 of 10 | 0.00195 | 0.0254 |
| S3 | -0.040 ± 0.007 | 0 of 10 | 0.00195 | 0.0254 |
| S1c | -0.025 ± 0.005 | 0 of 10 | 0.00195 | 0.0254 |
| S2c | -0.037 ± 0.005 | 0 of 10 | 0.00195 | 0.0254 |
| wine | -0.034 ± 0.008 | 0 of 10 | 0.00195 | 0.0254 |
| breast_cancer | -0.018 ± 0.006 | 0 of 10 | 0.00195 | 0.0254 |
| thyroid | -0.032 ± 0.009 | 0 of 10 | 0.00195 | 0.0254 |
| shuttle | -0.002 ± 0.011 | 4 of 10 | 0.695 | 1 |
| magic | -0.013 ± 0.011 | 1 of 10 | 0.0371 | 0.148 |
| letter | -0.136 ± 0.019 | 0 of 10 | 0.00195 | 0.0254 |
| landsat | +0.001 ± 0.015 | 5 of 10 | 1 | 1 |
| page_blocks | +0.000 ± 0.012 | 5 of 10 | 1 | 1 |

## PACE-SHAP-cond minus PACE-SHAP: interventional deletion margin AUC (secondary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for PACE-SHAP-cond: negative.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.038 ± 0.006 | 10 of 10 | 0.00195 | 0.0254 |
| S2 | +0.053 ± 0.009 | 10 of 10 | 0.00195 | 0.0254 |
| S3 | +0.046 ± 0.005 | 10 of 10 | 0.00195 | 0.0254 |
| S1c | +0.033 ± 0.007 | 10 of 10 | 0.00195 | 0.0254 |
| S2c | +0.037 ± 0.009 | 10 of 10 | 0.00195 | 0.0254 |
| wine | +0.072 ± 0.006 | 10 of 10 | 0.00195 | 0.0254 |
| breast_cancer | +0.087 ± 0.009 | 10 of 10 | 0.00195 | 0.0254 |
| thyroid | +0.035 ± 0.009 | 10 of 10 | 0.00195 | 0.0254 |
| shuttle | +0.031 ± 0.012 | 10 of 10 | 0.00195 | 0.0254 |
| magic | +0.103 ± 0.019 | 10 of 10 | 0.00195 | 0.0254 |
| letter | +0.095 ± 0.006 | 10 of 10 | 0.00195 | 0.0254 |
| landsat | +0.177 ± 0.022 | 10 of 10 | 0.00195 | 0.0254 |
| page_blocks | +0.036 ± 0.019 | 9 of 10 | 0.00391 | 0.0254 |

## PACE-SHAP-cond minus PACE-SHAP: interventional insertion margin AUC (secondary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for PACE-SHAP-cond: positive.

| Dataset | Difference | Seeds where PACE-SHAP-cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.035 ± 0.007 | 0 of 10 | 0.00195 | 0.0254 |
| S2 | -0.047 ± 0.014 | 0 of 10 | 0.00195 | 0.0254 |
| S3 | -0.036 ± 0.009 | 0 of 10 | 0.00195 | 0.0254 |
| S1c | -0.027 ± 0.007 | 0 of 10 | 0.00195 | 0.0254 |
| S2c | -0.034 ± 0.007 | 0 of 10 | 0.00195 | 0.0254 |
| wine | -0.087 ± 0.009 | 0 of 10 | 0.00195 | 0.0254 |
| breast_cancer | -0.108 ± 0.007 | 0 of 10 | 0.00195 | 0.0254 |
| thyroid | -0.040 ± 0.009 | 0 of 10 | 0.00195 | 0.0254 |
| shuttle | -0.036 ± 0.014 | 0 of 10 | 0.00195 | 0.0254 |
| magic | -0.110 ± 0.023 | 0 of 10 | 0.00195 | 0.0254 |
| letter | -0.195 ± 0.021 | 0 of 10 | 0.00195 | 0.0254 |
| landsat | -0.201 ± 0.019 | 0 of 10 | 0.00195 | 0.0254 |
| page_blocks | -0.050 ± 0.024 | 1 of 10 | 0.00391 | 0.0254 |

## PACE-flip-cond minus PACE-flip: conditional deletion margin AUC (secondary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for PACE-flip-cond: negative.

| Dataset | Difference | Seeds where PACE-flip-cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.005 ± 0.004 | 8 of 10 | 0.0488 | 0.244 |
| S2 | +0.005 ± 0.007 | 7 of 10 | 0.193 | 0.58 |
| S3 | +0.006 ± 0.006 | 7 of 10 | 0.0273 | 0.191 |
| S1c | +0.028 ± 0.007 | 10 of 10 | 0.00195 | 0.0254 |
| S2c | +0.028 ± 0.007 | 10 of 10 | 0.00195 | 0.0254 |
| wine | -0.000 ± 0.007 | 5 of 10 | 1 | 1 |
| breast_cancer | +0.067 ± 0.022 | 10 of 10 | 0.00195 | 0.0254 |
| thyroid | +0.009 ± 0.008 | 8 of 10 | 0.0273 | 0.191 |
| shuttle | -0.016 ± 0.013 | 2 of 10 | 0.00977 | 0.0781 |
| magic | -0.007 ± 0.009 | 3 of 10 | 0.105 | 0.422 |
| letter | -0.001 ± 0.007 | 3 of 10 | 0.695 | 1 |
| landsat | -0.084 ± 0.009 | 0 of 10 | 0.00195 | 0.0254 |
| page_blocks | -0.021 ± 0.011 | 1 of 10 | 0.00586 | 0.0527 |

## PACE-flip-cond minus PACE-flip: interventional deletion margin AUC (secondary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for PACE-flip-cond: negative.

| Dataset | Difference | Seeds where PACE-flip-cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.004 ± 0.006 | 7 of 10 | 0.16 | 0.48 |
| S2 | +0.005 ± 0.006 | 8 of 10 | 0.0645 | 0.258 |
| S3 | +0.007 ± 0.003 | 10 of 10 | 0.00195 | 0.0254 |
| S1c | +0.020 ± 0.006 | 10 of 10 | 0.00195 | 0.0254 |
| S2c | +0.025 ± 0.007 | 10 of 10 | 0.00195 | 0.0254 |
| wine | +0.019 ± 0.005 | 10 of 10 | 0.00195 | 0.0254 |
| breast_cancer | +0.054 ± 0.008 | 10 of 10 | 0.00195 | 0.0254 |
| thyroid | +0.009 ± 0.006 | 9 of 10 | 0.0137 | 0.0684 |
| shuttle | -0.002 ± 0.019 | 3 of 10 | 0.492 | 0.984 |
| magic | +0.024 ± 0.010 | 9 of 10 | 0.00391 | 0.0254 |
| letter | +0.021 ± 0.009 | 10 of 10 | 0.00195 | 0.0254 |
| landsat | +0.055 ± 0.016 | 10 of 10 | 0.00195 | 0.0254 |
| page_blocks | -0.005 ± 0.018 | 5 of 10 | 0.492 | 0.984 |

## Realism: off-manifold share of the inputs each scheme evaluates

An input is off-manifold if its nearest training row is farther than the 95th percentile of the training rows' own nearest-neighbour distances (standardised features), so about 5% of real points are. 200 sampled inputs per point and scheme; LIME's are weighted by its kernel. Lower = more realistic.

| Dataset | Interventional Shapley game (PACE-SHAP, KernelSHAP) | Conditional Shapley game (PACE-SHAP-cond, KernelSHAP-cond) | Quantile values (PACE-flip) | Neighbour values (PACE-flip-cond) | LIME's samples (kernel-weighted) | Interventional deletion/insertion test | Conditional deletion/insertion test | The explained points themselves |
|---|---|---|---|---|---|---|---|---|
| S1 | 0.038 ± 0.006 | 0.012 ± 0.003 | 0.049 ± 0.009 | 0.050 ± 0.010 | 0.130 ± 0.009 | 0.039 ± 0.005 | 0.020 ± 0.004 | 0.044 ± 0.015 |
| S2 | 0.042 ± 0.005 | 0.013 ± 0.003 | 0.052 ± 0.010 | 0.054 ± 0.011 | 0.128 ± 0.006 | 0.042 ± 0.006 | 0.022 ± 0.004 | 0.060 ± 0.018 |
| S3 | 0.041 ± 0.003 | 0.014 ± 0.003 | 0.055 ± 0.006 | 0.055 ± 0.008 | 0.133 ± 0.008 | 0.040 ± 0.003 | 0.022 ± 0.003 | 0.060 ± 0.017 |
| S1c | 0.137 ± 0.009 | 0.016 ± 0.004 | 0.110 ± 0.012 | 0.079 ± 0.013 | 0.352 ± 0.010 | 0.138 ± 0.010 | 0.031 ± 0.006 | 0.052 ± 0.022 |
| S2c | 0.139 ± 0.009 | 0.015 ± 0.003 | 0.115 ± 0.011 | 0.082 ± 0.012 | 0.358 ± 0.011 | 0.140 ± 0.007 | 0.033 ± 0.005 | 0.046 ± 0.019 |
| wine | 0.165 ± 0.022 | 0.014 ± 0.005 | 0.081 ± 0.021 | 0.064 ± 0.018 | 0.481 ± 0.061 | 0.161 ± 0.017 | 0.026 ± 0.007 | 0.047 ± 0.019 |
| breast_cancer | 0.127 ± 0.024 | 0.012 ± 0.008 | 0.040 ± 0.022 | 0.033 ± 0.019 | 0.624 ± 0.113 | 0.127 ± 0.024 | 0.018 ± 0.010 | 0.032 ± 0.018 |
| thyroid | 0.089 ± 0.014 | 0.026 ± 0.010 | 0.061 ± 0.024 | 0.056 ± 0.024 | 1.000 ± 0.000 | 0.088 ± 0.016 | 0.040 ± 0.012 | 0.048 ± 0.026 |
| shuttle | 0.706 ± 0.005 | 0.567 ± 0.011 | 0.661 ± 0.008 | 0.613 ± 0.008 | 1.000 ± 0.000 | 0.704 ± 0.004 | 0.574 ± 0.004 | 0.050 ± 0.022 |
| magic | 0.203 ± 0.015 | 0.024 ± 0.010 | 0.139 ± 0.022 | 0.059 ± 0.021 | 0.712 ± 0.047 | 0.201 ± 0.017 | 0.037 ± 0.012 | 0.040 ± 0.020 |
| letter | 0.627 ± 0.009 | 0.101 ± 0.010 | 0.319 ± 0.014 | 0.217 ± 0.017 | 0.995 ± 0.001 | 0.626 ± 0.013 | 0.262 ± 0.018 | 0.052 ± 0.017 |
| landsat | 0.671 ± 0.010 | 0.046 ± 0.013 | 0.263 ± 0.021 | 0.096 ± 0.022 | 1.000 ± 0.000 | 0.668 ± 0.012 | 0.084 ± 0.020 | 0.064 ± 0.027 |
| page_blocks | 0.296 ± 0.033 | 0.060 ± 0.009 | 0.235 ± 0.034 | 0.126 ± 0.024 | 0.997 ± 0.004 | 0.301 ± 0.035 | 0.093 ± 0.020 | 0.054 ± 0.017 |

## Conditional Shapley game (PACE-SHAP-cond, KernelSHAP-cond) minus Interventional Shapley game (PACE-SHAP, KernelSHAP): off-manifold share (primary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for shap_cond: negative.

| Dataset | Difference | Seeds where shap_cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.026 ± 0.005 | 0 of 10 | 0.00195 | 0.0254 |
| S2 | -0.029 ± 0.003 | 0 of 10 | 0.00195 | 0.0254 |
| S3 | -0.027 ± 0.002 | 0 of 10 | 0.00195 | 0.0254 |
| S1c | -0.122 ± 0.008 | 0 of 10 | 0.00195 | 0.0254 |
| S2c | -0.124 ± 0.007 | 0 of 10 | 0.00195 | 0.0254 |
| wine | -0.152 ± 0.020 | 0 of 10 | 0.00195 | 0.0254 |
| breast_cancer | -0.115 ± 0.019 | 0 of 10 | 0.00195 | 0.0254 |
| thyroid | -0.063 ± 0.008 | 0 of 10 | 0.00195 | 0.0254 |
| shuttle | -0.139 ± 0.011 | 0 of 10 | 0.00195 | 0.0254 |
| magic | -0.178 ± 0.009 | 0 of 10 | 0.00195 | 0.0254 |
| letter | -0.526 ± 0.006 | 0 of 10 | 0.00195 | 0.0254 |
| landsat | -0.624 ± 0.013 | 0 of 10 | 0.00195 | 0.0254 |
| page_blocks | -0.236 ± 0.027 | 0 of 10 | 0.00195 | 0.0254 |

## Neighbour values (PACE-flip-cond) minus Quantile values (PACE-flip): off-manifold share (secondary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for flip_cond: negative.

| Dataset | Difference | Seeds where flip_cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | +0.000 ± 0.002 | 5 of 10 | 0.91 | 1 |
| S2 | +0.002 ± 0.002 | 8 of 10 | 0.0645 | 0.193 |
| S3 | +0.001 ± 0.003 | 6 of 10 | 0.75 | 1 |
| S1c | -0.031 ± 0.006 | 0 of 10 | 0.00195 | 0.0254 |
| S2c | -0.033 ± 0.003 | 0 of 10 | 0.00195 | 0.0254 |
| wine | -0.017 ± 0.005 | 0 of 10 | 0.00195 | 0.0254 |
| breast_cancer | -0.007 ± 0.004 | 0 of 10 | 0.00195 | 0.0254 |
| thyroid | -0.005 ± 0.002 | 0 of 10 | 0.00195 | 0.0254 |
| shuttle | -0.048 ± 0.006 | 0 of 10 | 0.00195 | 0.0254 |
| magic | -0.080 ± 0.007 | 0 of 10 | 0.00195 | 0.0254 |
| letter | -0.103 ± 0.006 | 0 of 10 | 0.00195 | 0.0254 |
| landsat | -0.167 ± 0.016 | 0 of 10 | 0.00195 | 0.0254 |
| page_blocks | -0.109 ± 0.012 | 0 of 10 | 0.00195 | 0.0254 |

## Conditional deletion/insertion test minus Interventional deletion/insertion test: off-manifold share (secondary)

Paired over seeds; two-sided Wilcoxon signed-rank test, Holm across datasets. Better for test_cond: negative.

| Dataset | Difference | Seeds where test_cond is higher | p | p (Holm) |
|---|---|---|---|---|
| S1 | -0.019 ± 0.003 | 0 of 10 | 0.00195 | 0.0254 |
| S2 | -0.020 ± 0.002 | 0 of 10 | 0.00195 | 0.0254 |
| S3 | -0.018 ± 0.001 | 0 of 10 | 0.00195 | 0.0254 |
| S1c | -0.108 ± 0.007 | 0 of 10 | 0.00195 | 0.0254 |
| S2c | -0.107 ± 0.005 | 0 of 10 | 0.00195 | 0.0254 |
| wine | -0.135 ± 0.015 | 0 of 10 | 0.00195 | 0.0254 |
| breast_cancer | -0.109 ± 0.016 | 0 of 10 | 0.00195 | 0.0254 |
| thyroid | -0.048 ± 0.008 | 0 of 10 | 0.00195 | 0.0254 |
| shuttle | -0.130 ± 0.005 | 0 of 10 | 0.00195 | 0.0254 |
| magic | -0.164 ± 0.008 | 0 of 10 | 0.00195 | 0.0254 |
| letter | -0.364 ± 0.010 | 0 of 10 | 0.00195 | 0.0254 |
| landsat | -0.584 ± 0.023 | 0 of 10 | 0.00195 | 0.0254 |
| page_blocks | -0.208 ± 0.020 | 0 of 10 | 0.00195 | 0.0254 |

## Correlated distractor (S1c, S2c)

Feature 3 is a noisy copy of feature 0 that the true rule does not use. x3 share = mean over points of |attribution of x3| / sum of |attributions|. The model's own reliance: share of PACE-flip's single-feature decision flips (interventional) that come from x3.

| Dataset | Method | x3 share | Ground truth top-3 |
|---|---|---|---|
| S1c | PACE-SHAP | 0.201 ± 0.011 | 0.707 ± 0.018 |
| S1c | PACE-SHAP-cond | 0.175 ± 0.003 | 0.625 ± 0.018 |
| S1c | PACE-flip | 0.213 ± 0.015 | 0.556 ± 0.023 |
| S1c | PACE-flip-cond | 0.167 ± 0.012 | 0.541 ± 0.021 |
| S1c | KernelSHAP | 0.200 ± 0.011 | 0.697 ± 0.012 |
| S1c | KernelSHAP-cond | 0.170 ± 0.006 | 0.601 ± 0.016 |
| S1c | LIME | 0.217 ± 0.014 | 0.721 ± 0.015 |
| S1c | Random | 0.101 ± 0.007 | 0.307 ± 0.025 |
| S2c | PACE-SHAP | 0.176 ± 0.008 | 0.689 ± 0.021 |
| S2c | PACE-SHAP-cond | 0.154 ± 0.011 | 0.617 ± 0.023 |
| S2c | PACE-flip | 0.167 ± 0.013 | 0.479 ± 0.015 |
| S2c | PACE-flip-cond | 0.128 ± 0.007 | 0.467 ± 0.020 |
| S2c | KernelSHAP | 0.174 ± 0.008 | 0.693 ± 0.022 |
| S2c | KernelSHAP-cond | 0.149 ± 0.013 | 0.588 ± 0.021 |
| S2c | LIME | 0.203 ± 0.011 | 0.721 ± 0.010 |
| S2c | Random | 0.101 ± 0.007 | 0.307 ± 0.025 |

| Dataset | Flip mass on x3 | Flip mass on x0–x2 |
|---|---|---|
| S1c | 0.191 ± 0.019 | 0.511 ± 0.030 |
| S2c | 0.134 ± 0.009 | 0.402 ± 0.015 |

## Ground truth top-3 (synthetic sets)

| Method | S1 | S2 | S3 | S1c | S2c |
|---|---|---|---|---|---|
| PACE-SHAP | 0.86 | 0.82 | 0.81 | 0.71 | 0.69 |
| PACE-SHAP-cond | 0.72 | 0.67 | 0.66 | 0.63 | 0.62 |
| PACE-flip | 0.57 | 0.51 | 0.52 | 0.56 | 0.48 |
| PACE-flip-cond | 0.57 | 0.51 | 0.52 | 0.54 | 0.47 |
| KernelSHAP | 0.85 | 0.81 | 0.79 | 0.70 | 0.69 |
| KernelSHAP-cond | 0.69 | 0.64 | 0.62 | 0.60 | 0.59 |
| LIME | 0.93 | 0.92 | 0.90 | 0.72 | 0.72 |
| Random | 0.31 | 0.31 | 0.31 | 0.31 | 0.31 |

## Milliseconds per explanation

Measured with several runs sharing the CPU: compare within a dataset only.

| Dataset | PACE-SHAP | PACE-SHAP-cond | PACE-flip | PACE-flip-cond | KernelSHAP | KernelSHAP-cond | LIME | Random |
|---|---|---|---|---|---|---|---|---|
| S1 | 10075.3 | 10093.3 | 4.1 | 51.1 | 53609.5 | 53611.3 | 1120.6 | 0.0 |
| S2 | 8184.4 | 8199.8 | 3.4 | 42.4 | 56156.6 | 56158.4 | 1190.3 | 0.0 |
| S3 | 8000.6 | 8015.7 | 3.6 | 41.6 | 54061.1 | 54062.9 | 1150.0 | 0.0 |
| S1c | 6761.6 | 6776.3 | 3.2 | 35.9 | 46107.8 | 46109.6 | 1054.1 | 0.0 |
| S2c | 6623.7 | 6638.6 | 3.4 | 36.8 | 47548.7 | 47550.4 | 1019.2 | 0.0 |
| wine | 50.9 | 57.6 | 0.3 | 2.5 | 439.5 | 440.2 | 46.9 | 0.0 |
| breast_cancer | 694.9 | 734.8 | 1.5 | 6.3 | 4396.7 | 4398.7 | 113.8 | 0.0 |
| thyroid | 509.1 | 531.4 | 1.3 | 6.1 | 3734.5 | 3736.4 | 97.4 | 0.0 |
| shuttle | 40.7 | 45.2 | 0.1 | 1.8 | 363.3 | 364.6 | 20.6 | 0.0 |
| magic | 1046.8 | 1061.2 | 1.9 | 7.4 | 8906.9 | 8908.6 | 227.3 | 0.0 |
| letter | 2857.7 | 2902.2 | 4.4 | 21.3 | 28594.0 | 28595.5 | 709.4 | 0.0 |
| landsat | 3950.2 | 4153.5 | 3.6 | 20.2 | 18090.1 | 18092.1 | 484.7 | 0.0 |
| page_blocks | 367.5 | 391.0 | 0.6 | 4.5 | 1271.1 | 1273.4 | 53.2 | 0.0 |
