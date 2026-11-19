# Review Test 1 (descriptive): PACE-SHAP vs exact Shapley values computed in general ways

Rules: [`../PREREGISTRATION.md`](../PREREGISTRATION.md) (Test 1). v3's models and explainer rows. Milliseconds per explanation, measured with nothing else running.

## (a) Exact mode: the same values, computed three ways

| Dataset | d | Rows | Points | Largest difference: ExactExplainer | Largest difference: KernelSHAP, every coalition | Largest value | ms PACE-SHAP | ms ExactExplainer | ms KernelSHAP, every coalition | ExactExplainer / PACE-SHAP | KernelSHAP / PACE-SHAP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 10 | 200 | 30 | 1.6e-16 | 1.9e-16 | 0.083 | 2499.0 | 53707.6 | 54136.8 | 21.5× | 21.7× |
| S2 | 10 | 200 | 30 | 1.0e-16 | 2.1e-16 | 0.083 | 2359.2 | 52990.5 | 52775.8 | 22.5× | 22.4× |
| S3 | 10 | 200 | 30 | 1.3e-16 | 2.3e-16 | 0.113 | 2519.6 | 53775.9 | 57595.2 | 21.3× | 22.9× |
| shuttle | 9 | 200 | 30 | 2.5e-16 | 3.3e-16 | 0.129 | 19.2 | 477.7 | 505.9 | 24.8× | 26.3× |
| magic | 10 | 200 | 30 | 6.8e-17 | 6.0e-17 | 0.026 | 376.1 | 11435.5 | 10198.0 | 30.4× | 27.1× |
| page_blocks | 10 | 200 | 30 | 1.6e-16 | 1.4e-16 | 0.034 | 68.0 | 1121.3 | 1106.6 | 16.5× | 16.3× |

Largest difference: the largest absolute difference from PACE-SHAP over every point, feature and seed. Largest value: the largest absolute PACE-SHAP value, for scale.

## (b) Sampled mode: error against exact values

| Dataset | d | Orders per row | Points | Relative L1 error | Top-3 overlap | ms per explanation |
|---|---|---|---|---|---|---|
| wine | 13 | exact (reference) | 108 | 0 | 1 | 262.6 |
| wine | 13 | 8 | 108 | 0.0202 | 0.968 | 7.3 |
| wine | 13 | 16 | 108 | 0.0145 | 0.981 | 9.1 |
| wine | 13 | 32 | 108 | 0.0096 | 0.995 | 13.4 |
| wine | 13 | 64 | 108 | 0.0072 | 1.000 | 24.7 |
| letter | 16 | exact (reference) | 5 | 0 | 1 | 41655.4 |
| letter | 16 | 8 | 5 | 0.0281 | 1.000 | 635.5 |
| letter | 16 | 16 | 5 | 0.0237 | 1.000 | 746.8 |
| letter | 16 | 32 | 5 | 0.0170 | 1.000 | 909.4 |
| letter | 16 | 64 | 5 | 0.0099 | 1.000 | 1288.6 |

Relative L1 error: sum over features of |sampled - exact|, divided by the sum of |exact|, averaged over points. Top-3 overlap: Jaccard overlap of the top 3 features by signed value, as in `faithfulness_v3/settings.md`.
