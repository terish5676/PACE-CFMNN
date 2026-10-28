# Deletion faithfulness (lower AUC = more faithful)

50 test points per dataset and seed; seeds [0, 1, 2]; features deleted by copying values from real background rows (10 draws); main model (fixed centroid + search).

Stability = top-3 overlap (Jaccard) when the explainer is re-run with another seed (1 = identical). Ground truth = share of top-3 features that are truly relevant (synthetic sets only).

| Dataset | Method | Class-kept AUC | Margin AUC | Stability | Ground truth top-3 | ms / explanation |
|---|---|---|---|---|---|---|
| S1 | PACE-contrastive | 0.555 ± 0.012 | 0.093 ± 0.023 | 1.00 | 0.42 | 0.26 |
| S1 | PACE-multi | 0.536 ± 0.005 | 0.070 ± 0.013 | 1.00 | 0.49 | 0.50 |
| S1 | PACE-flip | 0.443 ± 0.011 | -0.107 ± 0.018 | 1.00 | 0.57 | 1.11 |
| S1 | Occlusion (brute force) | 0.443 ± 0.011 | -0.107 ± 0.018 | 1.00 | 0.57 | 86.20 |
| S1 | PACE-decomposition | 0.602 ± 0.013 | 0.163 ± 0.026 | 1.00 | 0.31 | 0.35 |
| S1 | KernelSHAP | 0.527 ± 0.009 | 0.051 ± 0.025 | 0.88 | 0.79 | 547.90 |
| S1 | LIME | 0.535 ± 0.003 | 0.074 ± 0.009 | 1.00 | 1.00 | 266.04 |
| S1 | Random | 0.664 ± 0.003 | 0.275 ± 0.009 | 0.23 | 0.29 | 0.00 |
| S2 | PACE-contrastive | 0.578 ± 0.027 | 0.139 ± 0.044 | 1.00 | 0.44 | 0.28 |
| S2 | PACE-multi | 0.574 ± 0.012 | 0.135 ± 0.020 | 1.00 | 0.46 | 0.52 |
| S2 | PACE-flip | 0.441 ± 0.011 | -0.110 ± 0.012 | 1.00 | 0.54 | 0.97 |
| S2 | Occlusion (brute force) | 0.441 ± 0.011 | -0.110 ± 0.012 | 1.00 | 0.54 | 86.08 |
| S2 | PACE-decomposition | 0.592 ± 0.016 | 0.151 ± 0.018 | 1.00 | 0.33 | 0.34 |
| S2 | KernelSHAP | 0.542 ± 0.020 | 0.083 ± 0.035 | 0.88 | 0.80 | 549.20 |
| S2 | LIME | 0.562 ± 0.028 | 0.119 ± 0.052 | 1.00 | 1.00 | 274.99 |
| S2 | Random | 0.655 ± 0.001 | 0.271 ± 0.008 | 0.23 | 0.29 | 0.00 |
| S3 | PACE-contrastive | 0.623 ± 0.021 | 0.214 ± 0.052 | 1.00 | 0.47 | 0.28 |
| S3 | PACE-multi | 0.605 ± 0.034 | 0.189 ± 0.061 | 1.00 | 0.48 | 0.49 |
| S3 | PACE-flip | 0.525 ± 0.038 | 0.040 ± 0.062 | 1.00 | 0.51 | 0.95 |
| S3 | Occlusion (brute force) | 0.525 ± 0.038 | 0.040 ± 0.062 | 1.00 | 0.51 | 85.32 |
| S3 | PACE-decomposition | 0.665 ± 0.028 | 0.271 ± 0.058 | 1.00 | 0.32 | 0.31 |
| S3 | KernelSHAP | 0.601 ± 0.040 | 0.174 ± 0.071 | 0.85 | 0.68 | 537.27 |
| S3 | LIME | 0.603 ± 0.044 | 0.182 ± 0.078 | 1.00 | 1.00 | 269.92 |
| S3 | Random | 0.704 ± 0.044 | 0.341 ± 0.083 | 0.23 | 0.29 | 0.00 |
| wine | PACE-contrastive | 0.535 ± 0.035 | 0.058 ± 0.049 | 1.00 | – | 0.02 |
| wine | PACE-multi | 0.523 ± 0.033 | 0.027 ± 0.047 | 1.00 | – | 0.03 |
| wine | PACE-flip | 0.488 ± 0.022 | -0.047 ± 0.041 | 1.00 | – | 0.11 |
| wine | Occlusion (brute force) | 0.488 ± 0.022 | -0.047 ± 0.041 | 1.00 | – | 5.68 |
| wine | PACE-decomposition | 0.523 ± 0.024 | 0.034 ± 0.036 | 1.00 | – | 0.04 |
| wine | KernelSHAP | 0.510 ± 0.023 | 0.002 ± 0.040 | 0.94 | – | 27.72 |
| wine | LIME | 0.524 ± 0.031 | 0.037 ± 0.051 | 0.84 | – | 12.55 |
| wine | Random | 0.649 ± 0.007 | 0.255 ± 0.015 | 0.17 | – | 0.00 |
| breast_cancer | PACE-contrastive | 0.632 ± 0.023 | 0.214 ± 0.025 | 1.00 | – | 0.03 |
| breast_cancer | PACE-multi | 0.615 ± 0.013 | 0.176 ± 0.005 | 1.00 | – | 0.05 |
| breast_cancer | PACE-flip | 0.604 ± 0.007 | 0.158 ± 0.003 | 1.00 | – | 0.26 |
| breast_cancer | Occlusion (brute force) | 0.604 ± 0.007 | 0.158 ± 0.003 | 1.00 | – | 21.50 |
| breast_cancer | PACE-decomposition | 0.649 ± 0.024 | 0.244 ± 0.009 | 1.00 | – | 0.07 |
| breast_cancer | KernelSHAP | 0.626 ± 0.017 | 0.198 ± 0.012 | 0.60 | – | 61.32 |
| breast_cancer | LIME | 0.631 ± 0.015 | 0.231 ± 0.010 | 0.74 | – | 24.46 |
| breast_cancer | Random | 0.761 ± 0.007 | 0.421 ± 0.009 | 0.05 | – | 0.00 |
| thyroid | PACE-contrastive | 0.907 ± 0.061 | 0.605 ± 0.088 | 1.00 | – | 0.02 |
| thyroid | PACE-multi | 0.900 ± 0.072 | 0.569 ± 0.109 | 1.00 | – | 0.05 |
| thyroid | PACE-flip | 0.880 ± 0.081 | 0.534 ± 0.107 | 1.00 | – | 0.23 |
| thyroid | Occlusion (brute force) | 0.880 ± 0.081 | 0.534 ± 0.108 | 1.00 | – | 15.39 |
| thyroid | PACE-decomposition | 0.919 ± 0.066 | 0.724 ± 0.124 | 1.00 | – | 0.05 |
| thyroid | KernelSHAP | 0.914 ± 0.072 | 0.606 ± 0.127 | 0.99 | – | 53.16 |
| thyroid | LIME | 0.921 ± 0.054 | 0.646 ± 0.099 | 0.70 | – | 24.74 |
| thyroid | Random | 0.939 ± 0.049 | 0.741 ± 0.103 | 0.10 | – | 0.00 |
| shuttle | PACE-contrastive | 0.705 ± 0.043 | 0.258 ± 0.086 | 1.00 | – | 0.01 |
| shuttle | PACE-multi | 0.692 ± 0.032 | 0.254 ± 0.074 | 1.00 | – | 0.04 |
| shuttle | PACE-flip | 0.674 ± 0.035 | 0.250 ± 0.077 | 1.00 | – | 0.07 |
| shuttle | Occlusion (brute force) | 0.674 ± 0.035 | 0.250 ± 0.077 | 1.00 | – | 4.86 |
| shuttle | PACE-decomposition | 0.678 ± 0.037 | 0.283 ± 0.078 | 1.00 | – | 0.02 |
| shuttle | KernelSHAP | 0.685 ± 0.031 | 0.293 ± 0.066 | 0.99 | – | 62.04 |
| shuttle | LIME | 0.679 ± 0.042 | 0.297 ± 0.077 | 0.86 | – | 17.51 |
| shuttle | Random | 0.775 ± 0.018 | 0.433 ± 0.052 | 0.22 | – | 0.00 |
