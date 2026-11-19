# Run A: repair budgets, gates and certificates

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). RQ3's split and SR arm, seeds 110–119 (10), gate: cached. Each comparison: paired two-sided Wilcoxon over seeds, Holm across datasets; better / level / worse for the first arm. Written by `scripts/run_repair_sensitivity.py`.

## Primary: each arm vs SR-default, test macro-F1

**SR-b100 vs SR-default**: better 0, level 5, worse 1

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | +0.00 | 1.0000 | 1.0000 | level |
| skin | -0.05 | 0.9219 | 1.0000 | level |
| magic | -0.36 | 0.3223 | 1.0000 | level |
| letter | -1.14 | 0.0020 | 0.0117 | worse |
| landsat | -0.14 | 0.6250 | 1.0000 | level |
| page_blocks | +0.12 | 0.5625 | 1.0000 | level |

**SR-b1000 vs SR-default**: better 1, level 5, worse 0

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | +0.00 | 1.0000 | 1.0000 | level |
| skin | -0.11 | 0.3223 | 1.0000 | level |
| magic | -0.32 | 0.4922 | 1.0000 | level |
| letter | +1.42 | 0.0020 | 0.0117 | better |
| landsat | -0.06 | 1.0000 | 1.0000 | level |
| page_blocks | +0.02 | 1.0000 | 1.0000 | level |

**SR-ball vs SR-default**: better 1, level 5, worse 0

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | +0.00 | 1.0000 | 1.0000 | level |
| skin | -0.66 | 0.0195 | 0.0977 | level |
| magic | -1.52 | 0.0488 | 0.1953 | level |
| letter | +2.09 | 0.0020 | 0.0117 | better |
| landsat | -0.06 | 1.0000 | 1.0000 | level |
| page_blocks | +0.02 | 1.0000 | 1.0000 | level |

**SR-gate-none vs SR-default**: better 0, level 0, worse 6

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | -4.59 | 0.0078 | 0.0234 | worse |
| skin | -3.12 | 0.0020 | 0.0117 | worse |
| magic | -0.85 | 0.0078 | 0.0234 | worse |
| letter | -0.99 | 0.0039 | 0.0195 | worse |
| landsat | -0.71 | 0.0488 | 0.0488 | worse |
| page_blocks | -10.29 | 0.0039 | 0.0195 | worse |

**SR-gate-strict vs SR-default**: better 0, level 3, worse 3

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | -3.02 | 0.0039 | 0.0195 | worse |
| skin | -2.54 | 0.0020 | 0.0117 | worse |
| magic | -1.57 | 0.1309 | 0.2617 | level |
| letter | -0.94 | 0.0039 | 0.0195 | worse |
| landsat | -0.94 | 0.0840 | 0.2520 | level |
| page_blocks | -0.46 | 0.3594 | 0.3594 | level |

**SR-gate-loose vs SR-default**: better 0, level 5, worse 1

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | +0.00 | 1.0000 | 1.0000 | level |
| skin | -0.25 | 0.1562 | 0.7812 | level |
| magic | -0.71 | 0.0078 | 0.0469 | worse |
| letter | -0.06 | 0.4922 | 1.0000 | level |
| landsat | -0.14 | 0.3223 | 1.0000 | level |
| page_blocks | -0.35 | 1.0000 | 1.0000 | level |

**SR-gate-cumulative vs SR-default**: better 0, level 4, worse 2

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | -2.38 | 0.0312 | 0.0938 | level |
| skin | -2.10 | 0.0020 | 0.0117 | worse |
| magic | -1.06 | 0.0137 | 0.0547 | level |
| letter | -0.47 | 0.0020 | 0.0117 | worse |
| landsat | -0.41 | 0.1309 | 0.2617 | level |
| page_blocks | -0.82 | 0.1934 | 0.2617 | level |

## Test macro-F1 (%), mean over seeds

| Dataset | B0 | SR-default | SR-b100 | SR-b1000 | SR-ball | SR-gate-none | SR-gate-strict | SR-gate-loose | SR-gate-cumulative |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 83.59 | 90.49 | 90.49 | 90.49 | 90.49 | 85.90 | 87.46 | 90.49 | 88.10 |
| landsat | 81.73 | 83.69 | 83.55 | 83.63 | 83.63 | 82.98 | 82.74 | 83.55 | 83.27 |
| page_blocks | 68.08 | 69.90 | 70.01 | 69.91 | 69.91 | 59.61 | 69.44 | 69.55 | 69.08 |
| magic | 71.78 | 74.37 | 74.01 | 74.05 | 72.85 | 73.52 | 72.80 | 73.66 | 73.31 |
| letter | 73.61 | 76.22 | 75.09 | 77.64 | 78.32 | 75.23 | 75.28 | 76.17 | 75.75 |
| skin | 93.26 | 98.46 | 98.41 | 98.35 | 97.81 | 95.34 | 95.92 | 98.21 | 96.36 |

## Test accuracy (%), mean over seeds

| Dataset | B0 | SR-default | SR-b100 | SR-b1000 | SR-ball | SR-gate-none | SR-gate-strict | SR-gate-loose | SR-gate-cumulative |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 96.79 | 98.03 | 98.03 | 98.03 | 98.03 | 97.12 | 97.46 | 98.03 | 97.50 |
| landsat | 84.50 | 85.89 | 85.90 | 85.84 | 85.84 | 85.57 | 85.20 | 85.84 | 85.79 |
| page_blocks | 90.84 | 91.88 | 92.03 | 91.90 | 91.90 | 89.54 | 91.62 | 91.62 | 91.55 |
| magic | 73.33 | 76.48 | 75.95 | 76.12 | 74.50 | 75.61 | 74.41 | 75.75 | 75.03 |
| letter | 73.92 | 76.47 | 75.35 | 77.84 | 78.50 | 75.53 | 75.56 | 76.41 | 76.03 |
| skin | 95.90 | 99.00 | 98.97 | 98.93 | 98.58 | 97.11 | 97.45 | 98.84 | 97.70 |

## Lowest class recall (%), mean over seeds

| Dataset | B0 | SR-default | SR-b100 | SR-b1000 | SR-ball | SR-gate-none | SR-gate-strict | SR-gate-loose | SR-gate-cumulative |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 54.19 | 79.90 | 79.90 | 79.90 | 79.90 | 64.82 | 69.49 | 79.90 | 71.69 |
| landsat | 46.63 | 58.98 | 55.94 | 58.82 | 58.82 | 51.79 | 52.72 | 57.36 | 53.21 |
| page_blocks | 49.55 | 51.67 | 51.67 | 51.67 | 51.67 | 14.70 | 51.06 | 51.67 | 49.55 |
| magic | 69.60 | 67.98 | 69.15 | 68.07 | 70.73 | 67.65 | 70.47 | 67.68 | 70.75 |
| letter | 44.91 | 50.89 | 48.32 | 52.25 | 52.44 | 46.63 | 47.32 | 51.44 | 48.92 |
| skin | 81.78 | 96.86 | 96.78 | 96.35 | 94.52 | 87.59 | 88.88 | 96.06 | 90.41 |

## Boxes at the end, mean over seeds

| Dataset | B0 | SR-default | SR-b100 | SR-b1000 | SR-ball | SR-gate-none | SR-gate-strict | SR-gate-loose | SR-gate-cumulative |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 46 | 45 | 45 | 45 | 45 | 45 | 46 | 45 | 45 |
| landsat | 151 | 150 | 151 | 150 | 150 | 150 | 151 | 150 | 150 |
| page_blocks | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 |
| magic | 231 | 231 | 231 | 230 | 225 | 231 | 231 | 231 | 231 |
| letter | 458 | 458 | 458 | 457 | 452 | 458 | 458 | 458 | 458 |
| skin | 80 | 79 | 80 | 79 | 79 | 79 | 80 | 79 | 79 |

## Training error at the end (%), mean over seeds

| Dataset | B0 | SR-default | SR-b100 | SR-b1000 | SR-ball | SR-gate-none | SR-gate-strict | SR-gate-loose | SR-gate-cumulative |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 1.97 | 0.39 | 0.39 | 0.39 | 0.39 | 1.24 | 0.94 | 0.39 | 0.90 |
| landsat | 9.06 | 6.32 | 7.12 | 6.32 | 6.32 | 7.24 | 5.93 | 6.77 | 6.88 |
| page_blocks | 8.30 | 7.32 | 7.48 | 7.32 | 7.32 | 9.04 | 7.41 | 7.76 | 7.47 |
| magic | 24.36 | 20.76 | 21.73 | 21.66 | 23.31 | 21.77 | 22.36 | 21.59 | 22.10 |
| letter | 20.80 | 16.84 | 18.91 | 14.33 | 13.20 | 18.26 | 17.92 | 17.24 | 17.51 |
| skin | 4.07 | 1.00 | 1.00 | 1.07 | 1.41 | 2.89 | 2.50 | 1.16 | 2.32 |

## Edits accepted, mean over seeds

| Dataset | B0 | SR-default | SR-b100 | SR-b1000 | SR-ball | SR-gate-none | SR-gate-strict | SR-gate-loose | SR-gate-cumulative |
|---|---|---|---|---|---|---|---|---|---|
| thyroid |  | 16 | 15 | 16 | 16 | 19 | 11 | 16 | 18 |
| landsat |  | 174 | 73 | 176 | 176 | 189 | 80 | 182 | 182 |
| page_blocks |  | 12 | 7 | 12 | 12 | 29 | 5 | 16 | 17 |
| magic |  | 197 | 79 | 479 | 817 | 198 | 48 | 198 | 185 |
| letter |  | 193 | 74 | 510 | 882 | 239 | 80 | 222 | 217 |
| skin |  | 36 | 21 | 58 | 103 | 41 | 12 | 37 | 41 |

## Test predictions fixed against B0, mean over seeds

| Dataset | B0 | SR-default | SR-b100 | SR-b1000 | SR-ball | SR-gate-none | SR-gate-strict | SR-gate-loose | SR-gate-cumulative |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 0.0 | 10.5 | 10.5 | 10.5 | 10.5 | 8.8 | 6.0 | 10.5 | 9.1 |
| landsat | 0.0 | 24.0 | 19.1 | 23.7 | 23.7 | 25.0 | 12.9 | 23.9 | 24.3 |
| page_blocks | 0.0 | 11.2 | 10.2 | 11.3 | 11.3 | 32.1 | 4.9 | 14.2 | 16.2 |
| magic | 0.0 | 217.4 | 141.3 | 274.9 | 292.8 | 217.9 | 53.5 | 218.9 | 193.0 |
| letter | 0.0 | 100.7 | 53.5 | 152.1 | 183.0 | 122.7 | 50.3 | 113.5 | 111.3 |
| skin | 0.0 | 926.5 | 868.1 | 946.0 | 955.3 | 867.1 | 411.7 | 919.7 | 894.5 |

## Test predictions broken against B0, mean over seeds

| Dataset | B0 | SR-default | SR-b100 | SR-b1000 | SR-ball | SR-gate-none | SR-gate-strict | SR-gate-loose | SR-gate-cumulative |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 0.0 | 1.6 | 1.6 | 1.6 | 1.6 | 6.4 | 1.2 | 1.6 | 4.0 |
| landsat | 0.0 | 15.1 | 10.1 | 15.1 | 15.1 | 18.1 | 8.4 | 15.3 | 16.0 |
| page_blocks | 0.0 | 5.5 | 3.7 | 5.5 | 5.5 | 39.2 | 0.6 | 9.9 | 12.3 |
| magic | 0.0 | 157.4 | 91.4 | 221.8 | 270.5 | 174.5 | 32.9 | 172.8 | 160.7 |
| letter | 0.0 | 49.6 | 24.7 | 73.6 | 91.2 | 90.3 | 17.3 | 63.6 | 68.9 |
| skin | 0.0 | 167.7 | 116.9 | 203.6 | 297.9 | 569.7 | 31.4 | 200.2 | 454.3 |

## Seconds, mean over seeds

| Dataset | B0 | SR-default | SR-b100 | SR-b1000 | SR-ball | SR-gate-none | SR-gate-strict | SR-gate-loose | SR-gate-cumulative |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| landsat | 0 | 65 | 2 | 3 | 3 | 3 | 4 | 3 | 3 |
| page_blocks | 0 | 2 | 1 | 2 | 2 | 0 | 2 | 2 | 2 |
| magic | 0 | 57 | 4 | 13 | 19 | 6 | 8 | 6 | 6 |
| letter | 0 | 124 | 7 | 28 | 48 | 12 | 13 | 12 | 13 |
| skin | 0 | 12 | 8 | 16 | 31 | 10 | 39 | 11 | 11 |

## Edits rejected or skipped, by reason (mean over seeds)

```
                                rej_macro_f1  rej_class_recall  rej_box_cap  skip_already_right  skip_not_inside
dataset     arm                                                                                                 
skin        SR-default                   7.6               1.9          0.0               251.7              2.4
            SR-b100                      4.8               0.8          0.0                73.2              0.2
            SR-b1000                    14.4               3.6          0.0               903.8             19.7
            SR-ball                     43.0               8.4          0.0              4697.9            130.9
            SR-gate-none                 0.0               0.0          0.0               249.4              9.8
            SR-gate-strict              35.6             151.2          0.0               100.6              0.2
            SR-gate-loose                5.2               2.0          0.0               252.6              3.2
            SR-gate-cumulative           1.4               0.4          0.0               251.8              5.8
letter      SR-default                   0.3              56.0          0.0                50.0              0.4
            SR-b100                      0.2              18.4          0.0                 7.1              0.0
            SR-b1000                     1.0             179.2          0.0               301.7              8.2
            SR-ball                      2.7             358.1          0.0               794.2             43.0
            SR-gate-none                 0.0               0.0          0.0                59.8              0.8
            SR-gate-strict             105.9              88.7          0.0                24.9              0.2
            SR-gate-loose                0.0              22.1          0.0                55.0              0.6
            SR-gate-cumulative           0.0              29.2          0.0                53.6              0.5
magic       SR-default                   2.4               1.3          0.0                98.5              1.0
            SR-b100                      1.4               0.6          0.0                18.7              0.0
            SR-b1000                     6.6               3.1          0.0               479.5             32.2
            SR-ball                     11.5               5.4          0.0              1275.6            206.8
            SR-gate-none                 0.0               0.0          0.0               100.1              1.5
            SR-gate-strict             115.6             109.1          0.0                27.0              0.0
            SR-gate-loose                0.3               0.1          0.0                99.7              1.5
            SR-gate-cumulative           1.1              28.4          0.0                83.8              1.5
landsat     SR-default                   1.2              17.5          0.0                89.6              3.8
            SR-b100                      0.5               6.5          0.0                19.9              0.2
            SR-b1000                     1.2              18.4          0.0                92.1              3.9
            SR-ball                      1.2              18.4          0.0                92.1              3.9
            SR-gate-none                 0.0               0.0          0.0                89.3              7.5
            SR-gate-strict              98.8              54.0          0.0                52.7              0.1
            SR-gate-loose                0.2               6.3          0.0                91.6              5.2
            SR-gate-cumulative           0.0               8.9          0.0                89.0              6.1
page_blocks SR-default                  52.8             110.1          0.0                44.5              0.5
            SR-b100                     24.9              50.5          0.0                17.2              0.2
            SR-b1000                    55.1             112.1          0.0                47.2              0.5
            SR-ball                     55.1             112.1          0.0                47.2              0.5
            SR-gate-none                 0.0               0.0          0.0               161.9             28.3
            SR-gate-strict              76.4             121.2          0.0                17.0              0.0
            SR-gate-loose               39.9             107.7          0.0                54.8              1.4
            SR-gate-cumulative          29.1             108.9          0.0                59.9              4.9
thyroid     SR-default                   4.4               1.6          0.0                49.2              0.2
            SR-b100                      4.4               1.6          0.0                47.9              0.2
            SR-b1000                     4.4               1.6          0.0                49.2              0.2
            SR-ball                      4.4               1.6          0.0                49.2              0.2
            SR-gate-none                 0.0               0.0          0.0                46.8              4.7
            SR-gate-strict               9.7              19.2          0.0                31.4              0.0
            SR-gate-loose                2.8               3.0          0.0                49.2              0.2
            SR-gate-cumulative           0.7               1.1          0.0                46.5              4.2
```

## Certificates on SR-default (100 test rows, every accepted edit)

Misses: rows whose prediction or explanation changed but the certificate did not flag them (sum over seeds; any miss is a failure). Size: share of rows flagged per edit (smaller is tighter). Precision: share of flagged rows that really changed (mean over seeds).

| Certificate | Misses | Size (% of rows per edit) | Precision (%) |
|---|---|---|---|
| loose | 0 | 12.95 | 2.76 |
| tight | 0 | 6.33 | 6.22 |
| centroid | 0 | 0.79 | 38.29 |
| flip | 0 | 55.40 | 23.33 |
| decomposition | 0 | 4.98 | 99.44 |

Edits scored: 6272. Rows changed (prediction / PACE-flip / PACE-decomposition, summed over edits): 900 / 50386 / 10748.

Per dataset (mean size %, then misses):

```
             size%                                      misses                                  
             loose  tight centroid   flip decomposition  loose tight centroid flip decomposition
dataset                                                                                         
thyroid      24.57   0.97     0.16  64.22          9.03    0.0   0.0      0.0  0.0           0.0
landsat       4.00   2.22     0.47  22.20          1.54    0.0   0.0      0.0  0.0           0.0
page_blocks  34.71  22.38     2.30  79.53         15.25    0.0   0.0      0.0  0.0           0.0
magic         8.55   7.81     0.70  56.44          1.89    0.0   0.0      0.0  0.0           0.0
letter        1.58   1.41     0.17  24.66          0.33    0.0   0.0      0.0  0.0           0.0
skin          4.31   3.20     0.92  85.38          1.81    0.0   0.0      0.0  0.0           0.0
```

