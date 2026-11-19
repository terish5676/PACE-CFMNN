# Run P8: a faster exact mode, and a fair brute-force baseline

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and 200 explainer rows, seeds 350, 351, 352, one task at a time. Milliseconds per explanation (mean over seeds), each including its explainer's set-up. Written by `scripts/run_recurrence.py`.

**Check: PASS.** The recurrence and the vectorised brute force match the current exact mode within 1e-12 on every point; on Wine and Letter both PACE-SHAP modes match shap's ExactExplainer within 1e-09.

| Dataset | d | Boxes | ms matmul | ms recurrence | ms brute force | ms ExactExplainer | matmul ÷ recurrence | brute force ÷ recurrence | Largest difference from matmul |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 10 | 1515 | 1869.1 | 1515.7 | 79525.4 | – | 1.23× | 52.5× | 9.7e-17 |
| S2 | 10 | 1537 | 1859.8 | 1513.4 | 79943.2 | – | 1.23× | 52.8× | 8.2e-17 |
| S3 | 10 | 1531 | 1743.3 | 1456.2 | 79880.0 | – | 1.20× | 54.9× | 8.0e-17 |
| shuttle | 9 | 14 | 8.3 | 7.1 | 336.5 | – | 1.17× | 47.3× | 1.9e-16 |
| magic | 10 | 285 | 219.1 | 211.7 | 15404.4 | – | 1.03× | 72.8× | 5.0e-17 |
| page_blocks | 10 | 25 | 27.8 | 24.9 | 1378.8 | – | 1.12× | 55.5× | 8.0e-17 |
| wine | 13 | 32 | 184.3 | 86.4 | 6258.5 | 2360.7 | 2.13× | 72.4× | 5.6e-16 |
| letter | 16 | 549 | 36463.2 | 26093.6 | 2931822.3 | 1036717.6 | 1.40× | 112.4× | 1.9e-16 |

Letter, recurrence: 26.1 s per explanation. The plan raises the exact limit to 16 features if this is at most 5 s: **no**.
