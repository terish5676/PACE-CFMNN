# Run P8: a faster exact mode, and a fair brute-force baseline

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and 200 explainer rows, seeds 350, 351, 352, one task at a time. Milliseconds per explanation (mean over seeds), each including its explainer's set-up. Written by `scripts/run_recurrence.py`.

**Check: FAIL.** The recurrence and the vectorised brute force match the current exact mode within 1e-12 on every point; on Wine and Letter both PACE-SHAP modes match shap's ExactExplainer within 1e-09.
- letter seed 350: PACE-SHAP, matmul differs from ExactExplainer by 3.9e-02
- letter seed 350: PACE-SHAP, recurrence differs from ExactExplainer by 3.9e-02

| Dataset | d | Boxes | ms matmul | ms recurrence | ms brute force | ms ExactExplainer | matmul ÷ recurrence | brute force ÷ recurrence | Largest difference from matmul |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 10 | 1515 | 1739.7 | 1501.7 | 79637.9 | – | 1.16× | 53.0× | 9.7e-17 |
| S2 | 10 | 1537 | 1738.7 | 1484.2 | 80462.9 | – | 1.17× | 54.2× | 8.2e-17 |
| S3 | 10 | 1531 | 1644.7 | 1424.8 | 79229.8 | – | 1.15× | 55.6× | 8.0e-17 |
| shuttle | 9 | 14 | 8.9 | 6.7 | 345.4 | – | 1.31× | 51.3× | 1.9e-16 |
| magic | 10 | 285 | 216.0 | 187.6 | 15480.4 | – | 1.15× | 82.5× | 5.0e-17 |
| page_blocks | 10 | 25 | 26.8 | 24.7 | 1388.4 | – | 1.08× | 56.2× | 8.0e-17 |
| wine | 13 | 32 | 176.1 | 87.1 | 6336.5 | 2362.7 | 2.02× | 72.7× | 5.6e-16 |
| letter | 16 | 549 | 36340.3 | 26443.9 | 2876665.8 | 1003330.2 | 1.37× | 108.8× | 1.9e-16 |

Letter, recurrence: 26.4 s per explanation. The plan raises the exact limit to 16 features if this is at most 5 s: **no**.
