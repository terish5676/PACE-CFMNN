# Run P1: time to a given accuracy, and one clean re-time

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and 200 explainer rows, one task at a time. Written by `scripts/run_time_to_accuracy.py`.

## Part (a): KernelSHAP's time to reach 1% of the exact values

Seeds 300, 301, 302, 10 points per seed. Relative L1 error against PACE-SHAP's exact values, median over points; b* is the smallest budget with median error at most 1%. Time ratio: KernelSHAP at b* over exact PACE-SHAP (mean over points, 95% bootstrap interval).

| Dataset | d | Median error at 100 | 200 | 400 | 700 | all | b* | ms PACE-SHAP (exact) | ms KernelSHAP at b* | Ratio at b* (95% interval) | Ratio at 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 10 | 13.97% | 9.04% | 6.40% | 4.03% | 0.00% | all | 1733 | 29233 | 16.9× (15.9–18.1) | 3.5× |
| S2 | 10 | 16.53% | 9.62% | 7.53% | 5.76% | 0.00% | all | 1774 | 30699 | 17.3× (16.0–18.9) | 3.3× |
| S3 | 10 | 20.43% | 12.59% | 9.11% | 7.10% | 0.00% | all | 1758 | 31722 | 18.0× (16.8–19.5) | 3.9× |
| shuttle | 9 | 2.39% | 1.87% | 1.33% | – | 0.00% | all | 9 | 187 | 20.1× (16.8–23.6) | 7.6× |
| magic | 10 | 22.59% | 13.51% | 8.95% | 7.25% | 0.00% | all | 219 | 5771 | 26.4× (24.6–28.0) | 6.0× |
| page_blocks | 10 | 13.02% | 9.34% | 5.68% | 4.58% | 0.00% | all | 29 | 599 | 20.7× (19.7–21.9) | 4.5× |

## Part (b): one clean re-time of the like-for-like comparison

Seed 303, 10 points per dataset, each explainer timed alone on the same 200 rows. KernelSHAP with 200 coalitions; PACE-SHAP in its default mode (exact for d ≤ 12, else 32 permutations).

| Dataset | d | PACE-SHAP mode | ms PACE-SHAP | ms KernelSHAP | KernelSHAP ÷ PACE-SHAP (95% interval) |
|---|---|---|---|---|---|
| S1 | 10 | exact | 1717 | 8158 | 4.8× (4.7–4.8) |
| S2 | 10 | exact | 1798 | 8668 | 4.8× (4.7–5.0) |
| S3 | 10 | exact | 1717 | 8460 | 4.9× (4.7–5.2) |
| S1c | 10 | exact | 1485 | 4933 | 3.3× (3.3–3.4) |
| S2c | 10 | exact | 1493 | 4964 | 3.3× (3.2–3.4) |
| wine | 13 | 32 permutations | 7 | 53 | 8.0× (7.8–8.3) |
| breast_cancer | 30 | 32 permutations | 84 | 417 | 4.9× (4.4–5.4) |
| thyroid | 21 | 32 permutations | 69 | 402 | 5.8× (5.6–6.0) |
| shuttle | 9 | exact | 10 | 74 | 7.7× (6.7–8.6) |
| magic | 10 | exact | 187 | 1134 | 6.1× (5.6–6.7) |
| letter | 16 | 32 permutations | 522 | 3946 | 7.6× (7.2–7.9) |
| landsat | 36 | 32 permutations | 596 | 1970 | 3.3× (3.2–3.5) |
| page_blocks | 10 | exact | 33 | 111 | 3.4× (3.1–3.8) |

Across the 13 datasets KernelSHAP took 3.3–8.0× as long as PACE-SHAP (median 4.9×).
