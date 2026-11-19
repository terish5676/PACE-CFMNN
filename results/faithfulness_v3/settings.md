# PACE-SHAP settings (approximation quality only, seed 99)

Reference: 200 explainer-background rows; exact for d <= 12, otherwise 128 orders per row. 20 test points per dataset. Faithfulness was not computed.

Rule (fixed before running): smallest R with every dataset at relative L1 error <= 0.1 and top-3 overlap >= 0.9 (at P = 16), then the smallest such P.

**Chosen: R = 200 background rows, P = 32 orders per row (d > 12).**

| Dataset | d | Mode | R | P | Rel. L1 error | Top-3 overlap | ms/explanation |
|---|---|---|---|---|---|---|---|
| S1 | 10 | exact | 20 |  | 0.273 | 0.73 | 301.0 |
| S1 | 10 | exact | 50 |  | 0.152 | 0.88 | 757.1 |
| S1 | 10 | exact | 100 |  | 0.090 | 0.95 | 1457.5 |
| S1 | 10 | exact | 200 |  | 0.000 | 1.00 | 2949.7 |
| S2 | 10 | exact | 20 |  | 0.708 | 0.78 | 276.0 |
| S2 | 10 | exact | 50 |  | 0.279 | 0.85 | 665.5 |
| S2 | 10 | exact | 100 |  | 0.117 | 0.93 | 1211.5 |
| S2 | 10 | exact | 200 |  | 0.000 | 1.00 | 2478.3 |
| S3 | 10 | exact | 20 |  | 0.190 | 0.91 | 239.1 |
| S3 | 10 | exact | 50 |  | 0.187 | 0.90 | 609.3 |
| S3 | 10 | exact | 100 |  | 0.237 | 0.93 | 1194.2 |
| S3 | 10 | exact | 200 |  | 0.000 | 1.00 | 2428.5 |
| wine | 13 | sampled | 20 | 2 | 0.324 | 0.80 | 2.5 |
| wine | 13 | sampled | 20 | 4 | 0.323 | 0.80 | 2.7 |
| wine | 13 | sampled | 20 | 8 | 0.320 | 0.82 | 3.1 |
| wine | 13 | sampled | 20 | 16 | 0.319 | 0.80 | 3.7 |
| wine | 13 | sampled | 50 | 2 | 0.064 | 0.93 | 4.1 |
| wine | 13 | sampled | 50 | 4 | 0.058 | 0.95 | 4.6 |
| wine | 13 | sampled | 50 | 8 | 0.055 | 0.95 | 5.4 |
| wine | 13 | sampled | 50 | 16 | 0.051 | 0.93 | 7.1 |
| wine | 13 | sampled | 100 | 2 | 0.035 | 0.97 | 5.5 |
| wine | 13 | sampled | 100 | 4 | 0.025 | 0.95 | 5.5 |
| wine | 13 | sampled | 100 | 8 | 0.020 | 1.00 | 6.6 |
| wine | 13 | sampled | 100 | 16 | 0.014 | 0.97 | 8.6 |
| wine | 13 | sampled | 200 | 2 | 0.035 | 0.97 | 5.0 |
| wine | 13 | sampled | 200 | 4 | 0.025 | 0.95 | 5.5 |
| wine | 13 | sampled | 200 | 8 | 0.020 | 1.00 | 6.7 |
| wine | 13 | sampled | 200 | 16 | 0.014 | 0.97 | 8.8 |
| breast_cancer | 30 | sampled | 20 | 2 | 0.290 | 0.69 | 5.2 |
| breast_cancer | 30 | sampled | 20 | 4 | 0.271 | 0.68 | 5.8 |
| breast_cancer | 30 | sampled | 20 | 8 | 0.267 | 0.75 | 6.3 |
| breast_cancer | 30 | sampled | 20 | 16 | 0.264 | 0.74 | 7.7 |
| breast_cancer | 30 | sampled | 50 | 2 | 0.179 | 0.83 | 11.2 |
| breast_cancer | 30 | sampled | 50 | 4 | 0.167 | 0.81 | 12.7 |
| breast_cancer | 30 | sampled | 50 | 8 | 0.162 | 0.88 | 14.2 |
| breast_cancer | 30 | sampled | 50 | 16 | 0.160 | 0.85 | 17.8 |
| breast_cancer | 30 | sampled | 100 | 2 | 0.112 | 0.85 | 22.3 |
| breast_cancer | 30 | sampled | 100 | 4 | 0.104 | 0.88 | 24.1 |
| breast_cancer | 30 | sampled | 100 | 8 | 0.102 | 0.85 | 28.0 |
| breast_cancer | 30 | sampled | 100 | 16 | 0.102 | 0.82 | 35.8 |
| breast_cancer | 30 | sampled | 200 | 2 | 0.044 | 0.93 | 87.0 |
| breast_cancer | 30 | sampled | 200 | 4 | 0.030 | 0.93 | 100.2 |
| breast_cancer | 30 | sampled | 200 | 8 | 0.021 | 0.95 | 140.0 |
| breast_cancer | 30 | sampled | 200 | 16 | 0.017 | 0.97 | 138.7 |
| thyroid | 21 | sampled | 20 | 2 | 0.751 | 0.45 | 3.1 |
| thyroid | 21 | sampled | 20 | 4 | 0.756 | 0.45 | 3.5 |
| thyroid | 21 | sampled | 20 | 8 | 0.741 | 0.45 | 4.2 |
| thyroid | 21 | sampled | 20 | 16 | 0.739 | 0.45 | 5.3 |
| thyroid | 21 | sampled | 50 | 2 | 0.498 | 0.50 | 7.3 |
| thyroid | 21 | sampled | 50 | 4 | 0.503 | 0.50 | 8.3 |
| thyroid | 21 | sampled | 50 | 8 | 0.493 | 0.51 | 10.5 |
| thyroid | 21 | sampled | 50 | 16 | 0.491 | 0.50 | 13.7 |
| thyroid | 21 | sampled | 100 | 2 | 0.230 | 0.65 | 16.5 |
| thyroid | 21 | sampled | 100 | 4 | 0.217 | 0.69 | 17.4 |
| thyroid | 21 | sampled | 100 | 8 | 0.214 | 0.70 | 20.3 |
| thyroid | 21 | sampled | 100 | 16 | 0.213 | 0.77 | 27.6 |
| thyroid | 21 | sampled | 200 | 2 | 0.052 | 0.82 | 54.7 |
| thyroid | 21 | sampled | 200 | 4 | 0.039 | 0.80 | 59.0 |
| thyroid | 21 | sampled | 200 | 8 | 0.029 | 0.91 | 66.2 |
| thyroid | 21 | sampled | 200 | 16 | 0.022 | 0.85 | 78.4 |
| shuttle | 9 | exact | 20 |  | 0.470 | 0.88 | 1.2 |
| shuttle | 9 | exact | 50 |  | 0.246 | 0.85 | 2.7 |
| shuttle | 9 | exact | 100 |  | 0.217 | 0.88 | 6.1 |
| shuttle | 9 | exact | 200 |  | 0.000 | 1.00 | 13.3 |
| magic | 10 | exact | 20 |  | 0.449 | 0.60 | 22.9 |
| magic | 10 | exact | 50 |  | 0.326 | 0.65 | 80.6 |
| magic | 10 | exact | 100 |  | 0.295 | 0.82 | 155.2 |
| magic | 10 | exact | 200 |  | 0.000 | 1.00 | 325.4 |
| letter | 16 | sampled | 20 | 2 | 0.289 | 0.80 | 54.4 |
| letter | 16 | sampled | 20 | 4 | 0.273 | 0.75 | 58.5 |
| letter | 16 | sampled | 20 | 8 | 0.248 | 0.80 | 58.6 |
| letter | 16 | sampled | 20 | 16 | 0.243 | 0.80 | 66.1 |
| letter | 16 | sampled | 50 | 2 | 0.183 | 0.85 | 138.2 |
| letter | 16 | sampled | 50 | 4 | 0.175 | 0.80 | 141.8 |
| letter | 16 | sampled | 50 | 8 | 0.157 | 0.85 | 149.1 |
| letter | 16 | sampled | 50 | 16 | 0.154 | 0.85 | 181.0 |
| letter | 16 | sampled | 100 | 2 | 0.109 | 0.90 | 289.0 |
| letter | 16 | sampled | 100 | 4 | 0.099 | 0.88 | 294.2 |
| letter | 16 | sampled | 100 | 8 | 0.091 | 0.85 | 343.0 |
| letter | 16 | sampled | 100 | 16 | 0.082 | 0.85 | 380.0 |
| letter | 16 | sampled | 200 | 2 | 0.052 | 1.00 | 606.9 |
| letter | 16 | sampled | 200 | 4 | 0.043 | 0.97 | 635.7 |
| letter | 16 | sampled | 200 | 8 | 0.028 | 0.95 | 685.8 |
| letter | 16 | sampled | 200 | 16 | 0.021 | 0.97 | 796.5 |
| landsat | 36 | sampled | 20 | 2 | 0.234 | 0.66 | 32.1 |
| landsat | 36 | sampled | 20 | 4 | 0.204 | 0.70 | 33.2 |
| landsat | 36 | sampled | 20 | 8 | 0.185 | 0.70 | 35.2 |
| landsat | 36 | sampled | 20 | 16 | 0.171 | 0.76 | 40.7 |
| landsat | 36 | sampled | 50 | 2 | 0.149 | 0.72 | 163.8 |
| landsat | 36 | sampled | 50 | 4 | 0.121 | 0.70 | 165.6 |
| landsat | 36 | sampled | 50 | 8 | 0.108 | 0.76 | 172.4 |
| landsat | 36 | sampled | 50 | 16 | 0.099 | 0.78 | 182.9 |
| landsat | 36 | sampled | 100 | 2 | 0.124 | 0.75 | 356.3 |
| landsat | 36 | sampled | 100 | 4 | 0.112 | 0.73 | 366.6 |
| landsat | 36 | sampled | 100 | 8 | 0.106 | 0.82 | 382.4 |
| landsat | 36 | sampled | 100 | 16 | 0.101 | 0.75 | 421.1 |
| landsat | 36 | sampled | 200 | 2 | 0.058 | 0.88 | 796.4 |
| landsat | 36 | sampled | 200 | 4 | 0.043 | 0.82 | 822.2 |
| landsat | 36 | sampled | 200 | 8 | 0.028 | 0.95 | 873.6 |
| landsat | 36 | sampled | 200 | 16 | 0.020 | 0.93 | 982.4 |
| page_blocks | 10 | exact | 20 |  | 0.478 | 0.49 | 5.2 |
| page_blocks | 10 | exact | 50 |  | 0.530 | 0.65 | 10.3 |
| page_blocks | 10 | exact | 100 |  | 0.254 | 0.66 | 29.7 |
| page_blocks | 10 | exact | 200 |  | 0.000 | 1.00 | 81.5 |
