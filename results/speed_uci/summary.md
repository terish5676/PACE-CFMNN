# PACE-flip speed

Same models, test points and background as `run_faithfulness.py` (main model: fixed centroid + search expansion; 50 test points per seed; seeds [0, 1, 2]; 32 replacement values per feature). Milliseconds per explained point, mean over seeds of the median of 5 runs (1 run for the old loop and occlusion, which are slow). All methods use the model's cache-friendly chunking. Times depend on the machine; compare within a row.

## Batch of points (one call explains all of them)

| Dataset | Boxes | PACE-flip, original loop | PACE-flip, new | New, tables prebuilt | PACE-multi | Occlusion (brute force) | New vs loop | New vs occlusion |
|---|---|---|---|---|---|---|---|---|
| magic | 285 | 2.93 | 0.634 | 0.569 | 0.135 | 24.9 | 5× | 39× |
| letter | 564 | 10.9 | 1.38 | 1.25 | 0.318 | 111.1 | 8× | 80× |
| landsat | 201 | 9.21 | 0.841 | 0.778 | 0.187 | 125.5 | 11× | 149× |
| page_blocks | 26 | 1.39 | 0.123 | 0.114 | 0.028 | 4.02 | 11× | 33× |

## One point per call (tables prebuilt for PACE-flip)

| Dataset | PACE-multi | PACE-flip, new |
|---|---|---|
| magic | 0.726 | 0.700 |
| letter | 1.16 | 1.11 |
| landsat | 0.747 | 0.803 |
| page_blocks | 0.584 | 0.428 |

## Work skipped and exactness

Boxes kept: per point and feature, boxes the exact bound could not rule out. Tie-break cells: share of (point, feature, value) cells where boxes of two classes tied at the top and the nearest centroid decided.

| Dataset | Boxes kept per feature | Boxes skipped | Tie-break cells | Identical to the original loop (all seeds) |
|---|---|---|---|---|
| magic | 36.0 of 285 | 87% | 60.3% | yes |
| letter | 46.7 of 564 | 92% | 49.9% | yes |
| landsat | 18.8 of 201 | 91% | 14.3% | yes |
| page_blocks | 6.2 of 26 | 76% | 71.1% | yes |
