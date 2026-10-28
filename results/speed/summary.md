# PACE-flip speed

Same models, test points and background as `run_faithfulness.py` (main model: fixed centroid + search expansion; 50 test points per seed; seeds [0, 1, 2]; 32 replacement values per feature). Milliseconds per explained point, mean over seeds of the median of 5 runs (1 run for the old loop and occlusion, which are slow). All methods use the model's cache-friendly chunking. Times depend on the machine; compare within a row.

## Batch of points (one call explains all of them)

| Dataset | Boxes | PACE-flip, original loop | PACE-flip, new | New, tables prebuilt | PACE-multi | Occlusion (brute force) | New vs loop | New vs occlusion |
|---|---|---|---|---|---|---|---|---|
| S1 | 1514 | 14.5 | 1.13 | 0.874 | 0.558 | 97.1 | 13× | 86× |
| S2 | 1532 | 14.2 | 1.10 | 0.817 | 0.611 | 97.2 | 13× | 89× |
| S3 | 1529 | 12.1 | 0.990 | 0.766 | 0.510 | 91.1 | 12× | 92× |
| wine | 31 | 1.92 | 0.101 | 0.076 | 0.032 | 5.80 | 19× | 57× |
| breast_cancer | 43 | 4.35 | 0.272 | 0.223 | 0.049 | 22.7 | 16× | 84× |
| thyroid | 55 | 3.76 | 0.250 | 0.247 | 0.050 | 16.5 | 15× | 66× |
| shuttle | 14 | 1.51 | 0.053 | 0.039 | 0.022 | 3.92 | 28× | 73× |

## One point per call (tables prebuilt for PACE-flip)

| Dataset | PACE-multi | PACE-flip, new |
|---|---|---|
| S1 | 1.20 | 1.11 |
| S2 | 1.14 | 1.06 |
| S3 | 1.11 | 1.04 |
| wine | 0.618 | 0.358 |
| breast_cancer | 0.700 | 0.490 |
| thyroid | 0.676 | 0.567 |
| shuttle | 0.748 | 0.388 |

## Work skipped and exactness

Boxes kept: per point and feature, boxes the exact bound could not rule out. Tie-break cells: share of (point, feature, value) cells where boxes of two classes tied at the top and the nearest centroid decided.

| Dataset | Boxes kept per feature | Boxes skipped | Tie-break cells | Identical to the original loop (all seeds) |
|---|---|---|---|---|
| S1 | 29.5 of 1514 | 98% | 0.0% | yes |
| S2 | 28.8 of 1532 | 98% | 0.0% | yes |
| S3 | 26.8 of 1529 | 98% | 0.0% | yes |
| wine | 8.9 of 31 | 71% | 0.0% | yes |
| breast_cancer | 8.1 of 43 | 81% | 0.4% | yes |
| thyroid | 11.4 of 55 | 79% | 6.4% | yes |
| shuttle | 3.2 of 14 | 77% | 13.7% | yes |
