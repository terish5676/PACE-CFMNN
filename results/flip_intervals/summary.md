# Run P7: exact flip sets against dense occlusion

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). v3's models and explainer rows, seeds 340–349 (10 seeds). Written by `scripts/run_flip_intervals.py`.

**Check: PASS.** On 17,500 (point, feature) pairs, every value of a 10,001-value grid that changes the class lies in the exact flip set (within 1e-09), every exact interval wider than the grid step contains a grid value that changes the class, and on all 85,680 pairs the class at PACE-flip's 32 values agrees with the exact set (2,414 values within 1e-09 of a breakpoint, a tie up to rounding, left out of that comparison).

| Dataset | d | Boxes | Pairs | Pairs with a flip | Missed by the 32 values | Missed share | Intervals per flipping pair | Median smallest change, exact | Median smallest change, 32 values | Points the 32 values call unflippable |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 10 | 1516 | 5,000 | 2,238 | 37 | 1.7% | 1.30 | 0.179 | 0.196 | 0 of 482 |
| S2 | 10 | 1534 | 5,000 | 2,559 | 45 | 1.8% | 1.34 | 0.139 | 0.155 | 2 of 487 |
| S3 | 10 | 1524 | 5,000 | 2,263 | 42 | 1.9% | 1.30 | 0.136 | 0.161 | 1 of 490 |
| wine | 13 | 31 | 4,680 | 336 | 18 | 5.4% | 1.10 | 0.227 | 0.239 | 0 of 98 |
| breast_cancer | 30 | 41 | 15,000 | 1,218 | 87 | 7.1% | 1.11 | 0.216 | 0.228 | 4 of 113 |
| thyroid | 21 | 54 | 10,500 | 1,282 | 578 | 45.1% | 1.24 | 0.011 | 0.014 | 128 of 446 |
| shuttle | 9 | 14 | 4,500 | 3,064 | 1,722 | 56.2% | 1.31 | 0.067 | 0.122 | 0 of 499 |
| magic | 10 | 285 | 5,000 | 4,285 | 597 | 13.9% | 1.99 | 0.021 | 0.030 | 4 of 488 |
| letter | 16 | 564 | 8,000 | 5,895 | 492 | 8.3% | 1.64 | 0.030 | 0.067 | 4 of 482 |
| landsat | 36 | 198 | 18,000 | 7,515 | 607 | 8.1% | 1.48 | 0.066 | 0.081 | 4 of 331 |
| page_blocks | 10 | 26 | 5,000 | 4,177 | 2,131 | 51.0% | 1.36 | 0.004 | 0.012 | 25 of 495 |

Over all datasets the 32 values miss 6,356 of 34,832 flipping pairs (18.2%). Smallest change: the distance from x_i to the nearest value of one feature that changes the class, minimised over features (points with a flip only).
