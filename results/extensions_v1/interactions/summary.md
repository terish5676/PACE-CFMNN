# RQ1 extension B4: PACE-SHAP interaction values

Exact SHAP interaction values of the predicted class's margin (PaceShap.interactions), 200 explainer rows, 50 test points per dataset and seed (fewer if the test split is smaller); seeds [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]. Pair strength = mean over points of |interaction value|. Mean ± 95% CI over seeds.

| Dataset | Most frequent top pair | Seeds where it is top | Its strength | Share of pair strength among x0-x2 |
|---|---|---|---|---|
| S1 | (x0, x1) | 10 of 10 | 0.0051 ± 0.0005 | 0.10 ± 0.01 |
| S2 | (x0, x1) | 10 of 10 | 0.0041 ± 0.0003 | 0.08 ± 0.00 |
| S3 | (x0, x1) | 10 of 10 | 0.0078 ± 0.0006 | 0.13 ± 0.00 |
| S1c | (x0, x3) | 10 of 10 | 0.0068 ± 0.0006 | 0.08 ± 0.00 |
| S2c | (x0, x3) | 10 of 10 | 0.0060 ± 0.0004 | 0.06 ± 0.00 |
| shuttle | (x0, x6) | 10 of 10 | 0.0011 ± 0.0002 | – |
| magic | (x0, x2) | 5 of 10 | 0.0011 ± 0.0002 | – |
| page_blocks | (x1, x4) | 2 of 10 | 0.0005 ± 0.0002 | – |

## S3: is the AND pair (x0, x1) the top pair? (primary)

(x0, x1) is the top pair in 10 of 10 seeds (pre-registered: supported if at least 8 of 10).

| Pair | Strength |
|---|---|
| (x0, x1) | 0.0078 ± 0.0006 |
| (x0, x2) | 0.0034 ± 0.0003 |
| (x1, x2) | 0.0037 ± 0.0003 |
| strongest pair with an irrelevant feature | 0.0032 ± 0.0001 |
