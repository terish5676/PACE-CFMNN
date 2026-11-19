# Self-repair at the theta chosen by validation

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The readable-baselines split, seeds 70–89 (20). Each comparison: paired two-sided Wilcoxon over seeds, Holm across datasets; better / level / worse for the first model. Written by `scripts/run_repair_theta.py`.

**Check:** 520 of 520 B0 rows reproduce the readable-baselines accuracy and macro-F1 exactly.

## Test accuracy (%), mean over seeds

| Dataset | B0-0.3 | SR-0.3 | B0-val | SR-val |
|---|---|---|---|---|
| S1 | 89.12 | 89.12 | 90.64 | 90.66 |
| S2 | 80.83 | 80.83 | 81.59 | 81.59 |
| S3 | 87.80 | 87.80 | 88.07 | 88.10 |
| wine | 98.33 | 98.33 | 98.19 | 98.19 |
| breast_cancer | 95.44 | 95.66 | 95.92 | 96.01 |
| digits | 96.01 | 96.01 | 95.83 | 95.83 |
| thyroid | 97.36 | 98.44 | 97.38 | 98.93 |
| shuttle | 99.78 | 99.84 | 99.94 | 99.95 |
| skin | 95.98 | 99.34 | 99.77 | 99.91 |
| magic | 71.64 | 76.71 | 81.76 | 81.76 |
| letter | 75.15 | 77.59 | 95.12 | 95.11 |
| landsat | 85.12 | 86.34 | 90.03 | 90.04 |
| page_blocks | 85.04 | 87.56 | 95.37 | 95.63 |

## Test macro-F1 (%), mean over seeds

| Dataset | B0-0.3 | SR-0.3 | B0-val | SR-val |
|---|---|---|---|---|
| S1 | 89.12 | 89.12 | 90.63 | 90.65 |
| S2 | 80.81 | 80.81 | 81.57 | 81.58 |
| S3 | 86.49 | 86.49 | 86.99 | 87.02 |
| wine | 98.35 | 98.35 | 98.25 | 98.25 |
| breast_cancer | 95.07 | 95.30 | 95.61 | 95.70 |
| digits | 95.99 | 95.99 | 95.82 | 95.82 |
| thyroid | 86.98 | 92.35 | 86.86 | 94.75 |
| shuttle | 86.23 | 92.40 | 94.18 | 94.98 |
| skin | 93.43 | 98.99 | 99.66 | 99.87 |
| magic | 70.38 | 74.69 | 79.56 | 79.56 |
| letter | 74.92 | 77.44 | 95.10 | 95.10 |
| landsat | 82.31 | 83.97 | 88.37 | 88.37 |
| page_blocks | 62.20 | 65.50 | 77.30 | 78.62 |

## Comparisons by accuracy

**P1 (primary): SR-0.3 vs B0-val** (accuracy). Does repair at theta = 0.3 match choosing theta by validation? better 1, level 5, worse 7

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -1.51 | 0.0005 | 0.0030 | worse |
| S2 | -0.76 | 0.0825 | 0.4126 | level |
| S3 | -0.26 | 0.6579 | 1.0000 | level |
| wine | +0.14 | 0.5099 | 1.0000 | level |
| breast_cancer | -0.26 | 0.5303 | 1.0000 | level |
| digits | +0.18 | 0.5828 | 1.0000 | level |
| thyroid | +1.07 | 0.0002 | 0.0011 | better |
| shuttle | -0.09 | 0.0001 | 0.0009 | worse |
| skin | -0.43 | 0.0000 | 0.0000 | worse |
| magic | -5.05 | 0.0000 | 0.0000 | worse |
| letter | -17.53 | 0.0001 | 0.0009 | worse |
| landsat | -3.69 | 0.0001 | 0.0009 | worse |
| page_blocks | -7.81 | 0.0000 | 0.0000 | worse |

**P2 (primary): SR-val vs B0-val** (accuracy). Does repair still help at the theta validation chose? better 3, level 10, worse 0

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | +0.02 | 0.2763 | 1.0000 | level |
| S2 | +0.01 | 0.7630 | 1.0000 | level |
| S3 | +0.03 | 0.6733 | 1.0000 | level |
| wine | +0.00 | 1.0000 | 1.0000 | level |
| breast_cancer | +0.09 | 0.1573 | 1.0000 | level |
| digits | +0.00 | 1.0000 | 1.0000 | level |
| thyroid | +1.55 | 0.0001 | 0.0011 | better |
| shuttle | +0.01 | 0.0233 | 0.2327 | level |
| skin | +0.14 | 0.0000 | 0.0005 | better |
| magic | -0.00 | 0.8631 | 1.0000 | level |
| letter | -0.00 | 0.6664 | 1.0000 | level |
| landsat | +0.00 | 0.3173 | 1.0000 | level |
| page_blocks | +0.26 | 0.0031 | 0.0340 | better |

**S1: SR-0.3 vs B0-0.3** (accuracy). Does repair help on this split? better 7, level 6, worse 0

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | +0.00 | 1.0000 | 1.0000 | level |
| S2 | +0.00 | 1.0000 | 1.0000 | level |
| S3 | +0.00 | 1.0000 | 1.0000 | level |
| wine | +0.00 | 1.0000 | 1.0000 | level |
| breast_cancer | +0.22 | 0.1308 | 0.7848 | level |
| digits | +0.00 | 1.0000 | 1.0000 | level |
| thyroid | +1.08 | 0.0001 | 0.0009 | better |
| shuttle | +0.07 | 0.0017 | 0.0119 | better |
| skin | +3.36 | 0.0000 | 0.0000 | better |
| magic | +5.07 | 0.0000 | 0.0000 | better |
| letter | +2.44 | 0.0000 | 0.0000 | better |
| landsat | +1.22 | 0.0001 | 0.0013 | better |
| page_blocks | +2.52 | 0.0003 | 0.0023 | better |

**S2: SR-val vs SR-0.3** (accuracy). With repair on both, which theta is better? better 8, level 5, worse 0

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | +1.53 | 0.0005 | 0.0034 | better |
| S2 | +0.76 | 0.0858 | 0.4289 | level |
| S3 | +0.29 | 0.6291 | 1.0000 | level |
| wine | -0.14 | 0.5099 | 1.0000 | level |
| breast_cancer | +0.35 | 0.4207 | 1.0000 | level |
| digits | -0.18 | 0.5828 | 1.0000 | level |
| thyroid | +0.49 | 0.0006 | 0.0036 | better |
| shuttle | +0.10 | 0.0001 | 0.0011 | better |
| skin | +0.57 | 0.0000 | 0.0000 | better |
| magic | +5.05 | 0.0001 | 0.0011 | better |
| letter | +17.53 | 0.0001 | 0.0011 | better |
| landsat | +3.70 | 0.0001 | 0.0011 | better |
| page_blocks | +8.07 | 0.0001 | 0.0011 | better |

## Comparisons by macro-F1 (S3)

**S3, P1: SR-0.3 vs B0-val** (macro_f1). Does repair at theta = 0.3 match choosing theta by validation? better 1, level 6, worse 6

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -1.51 | 0.0002 | 0.0012 | worse |
| S2 | -0.76 | 0.0826 | 0.4953 | level |
| S3 | -0.50 | 0.4209 | 1.0000 | level |
| wine | +0.10 | 0.6725 | 1.0000 | level |
| breast_cancer | -0.31 | 0.4997 | 1.0000 | level |
| digits | +0.18 | 0.5303 | 1.0000 | level |
| thyroid | +5.49 | 0.0000 | 0.0001 | better |
| shuttle | -1.78 | 0.2611 | 1.0000 | level |
| skin | -0.66 | 0.0000 | 0.0000 | worse |
| magic | -4.88 | 0.0000 | 0.0000 | worse |
| letter | -17.66 | 0.0000 | 0.0000 | worse |
| landsat | -4.40 | 0.0000 | 0.0000 | worse |
| page_blocks | -11.80 | 0.0000 | 0.0000 | worse |

**S3, P2: SR-val vs B0-val** (macro_f1). Does repair still help at the theta validation chose? better 3, level 10, worse 0

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | +0.02 | 0.1380 | 1.0000 | level |
| S2 | +0.01 | 0.8785 | 1.0000 | level |
| S3 | +0.03 | 0.3454 | 1.0000 | level |
| wine | +0.00 | 1.0000 | 1.0000 | level |
| breast_cancer | +0.10 | 0.1797 | 1.0000 | level |
| digits | +0.00 | 1.0000 | 1.0000 | level |
| thyroid | +7.89 | 0.0000 | 0.0000 | better |
| shuttle | +0.80 | 0.0074 | 0.0740 | level |
| skin | +0.21 | 0.0000 | 0.0004 | better |
| magic | +0.00 | 0.7353 | 1.0000 | level |
| letter | -0.00 | 0.6784 | 1.0000 | level |
| landsat | +0.00 | 0.3173 | 1.0000 | level |
| page_blocks | +1.32 | 0.0007 | 0.0078 | better |

**S3, S1: SR-0.3 vs B0-0.3** (macro_f1). Does repair help on this split? better 7, level 6, worse 0

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | +0.00 | 1.0000 | 1.0000 | level |
| S2 | +0.00 | 1.0000 | 1.0000 | level |
| S3 | +0.00 | 1.0000 | 1.0000 | level |
| wine | +0.00 | 1.0000 | 1.0000 | level |
| breast_cancer | +0.23 | 0.1380 | 0.8281 | level |
| digits | +0.00 | 1.0000 | 1.0000 | level |
| thyroid | +5.36 | 0.0000 | 0.0000 | better |
| shuttle | +6.17 | 0.0000 | 0.0000 | better |
| skin | +5.57 | 0.0000 | 0.0000 | better |
| magic | +4.30 | 0.0000 | 0.0000 | better |
| letter | +2.51 | 0.0000 | 0.0000 | better |
| landsat | +1.66 | 0.0000 | 0.0000 | better |
| page_blocks | +3.30 | 0.0000 | 0.0001 | better |

**S3, S2: SR-val vs SR-0.3** (macro_f1). With repair on both, which theta is better? better 7, level 6, worse 0

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | +1.53 | 0.0001 | 0.0011 | better |
| S2 | +0.77 | 0.0826 | 0.4953 | level |
| S3 | +0.53 | 0.4209 | 1.0000 | level |
| wine | -0.10 | 0.6725 | 1.0000 | level |
| breast_cancer | +0.41 | 0.3259 | 1.0000 | level |
| digits | -0.18 | 0.5303 | 1.0000 | level |
| thyroid | +2.41 | 0.0003 | 0.0023 | better |
| shuttle | +2.59 | 0.1231 | 0.6155 | level |
| skin | +0.87 | 0.0000 | 0.0000 | better |
| magic | +4.88 | 0.0000 | 0.0000 | better |
| letter | +17.66 | 0.0000 | 0.0000 | better |
| landsat | +4.40 | 0.0000 | 0.0000 | better |
| page_blocks | +13.12 | 0.0000 | 0.0000 | better |

## Descriptive

Theta chosen by validation (count over seeds):

- S1: 0.4 ×3, 0.5 ×17
- S2: 0.4 ×9, 0.5 ×11
- S3: 0.3 ×1, 0.4 ×13, 0.5 ×6
- wine: 0.05 ×1, 0.15 ×2, 0.2 ×6, 0.3 ×4, 0.4 ×4, 0.5 ×3
- breast_cancer: 0.05 ×2, 0.1 ×4, 0.15 ×7, 0.2 ×5, 0.3 ×2
- digits: 0.15 ×4, 0.2 ×8, 0.3 ×8
- thyroid: 0.3 ×2, 0.4 ×7, 0.5 ×11
- shuttle: 0.05 ×20
- skin: 0.05 ×20
- magic: 0.05 ×20
- letter: 0.05 ×20
- landsat: 0.05 ×13, 0.1 ×7
- page_blocks: 0.05 ×19, 0.1 ×1

Means over seeds (boxes; conditions per box; training errors; of them, inside the winning wrong-class box; of them, with a full box of their own class; edits accepted; test predictions fixed and broken against that theta's B0; seconds):

```
                        boxes  conditions_per_box  train_errors  train_errors_in_winning_box  train_errors_own_box_full  edits_accepted  test_fixed  test_broken  seconds
S1            B0-0.3  1516.20               10.00          0.00                         0.00                       0.00             NaN        0.00         0.00     1.43
              SR-0.3  1516.20               10.00          0.00                         0.00                       0.00            0.00        0.00         0.00    13.94
              B0-val   788.50               10.00         15.00                        15.00                      15.00             NaN        0.00         0.00     1.00
              SR-val   788.50               10.00          0.70                         0.55                       0.35           14.75        0.25         0.05    42.80
S2            B0-0.3  1533.30               10.00          0.25                         0.25                       0.25             NaN        0.00         0.00     1.42
              SR-0.3  1533.30               10.00          0.00                         0.00                       0.00            0.25        0.00         0.00    15.52
              B0-val   924.55               10.00         37.70                        37.70                      37.70             NaN        0.00         0.00     1.06
              SR-val   924.55               10.00          6.85                         4.00                       3.25           36.60        0.70         0.65   101.85
S3            B0-0.3  1525.35               10.00          0.05                         0.05                       0.05             NaN        0.00         0.00     1.59
              SR-0.3  1525.35               10.00          0.00                         0.00                       0.00            0.05        0.00         0.00    14.19
              B0-val  1032.15               10.00         14.05                        14.05                      14.05             NaN        0.00         0.00     1.26
              SR-val  1032.15               10.00          1.40                         0.90                       0.85           13.65        0.50         0.20    42.06
wine          B0-0.3    31.35               13.00          0.00                         0.00                       0.00             NaN        0.00         0.00     0.02
              SR-0.3    31.35               13.00          0.00                         0.00                       0.00            0.00        0.00         0.00     0.10
              B0-val    41.10               12.99          0.00                         0.00                       0.00             NaN        0.00         0.00     0.02
              SR-val    41.10               12.99          0.00                         0.00                       0.00            0.00        0.00         0.00     0.12
breast_cancer B0-0.3    42.75               29.99          3.50                         3.50                       3.50             NaN        0.00         0.00     0.06
              SR-0.3    42.75               29.99          1.20                         0.90                       0.70            3.30        0.30         0.05     0.35
              B0-val   140.35               30.00          0.75                         0.75                       0.75             NaN        0.00         0.00     0.08
              SR-val   140.35               30.00          0.35                         0.15                       0.10            0.70        0.10         0.00     0.59
digits        B0-0.3   294.35               62.42          0.00                         0.00                       0.00             NaN        0.00         0.00     0.26
              SR-0.3   294.35               62.42          0.00                         0.00                       0.00            0.00        0.00         0.00     4.58
              B0-val   431.65               63.20          0.00                         0.00                       0.00             NaN        0.00         0.00     0.30
              SR-val   431.65               63.20          0.00                         0.00                       0.00            0.00        0.00         0.00     6.56
thyroid       B0-0.3    53.85               16.62         94.95                        94.95                      94.95             NaN        0.00         0.00     0.62
              SR-0.3    53.30               16.93         24.80                        16.70                      10.70           18.10       17.85         2.25    14.86
              B0-val    21.95               14.71        121.40                       121.40                     121.40             NaN        0.00         0.00     0.51
              SR-val    21.00               15.31         33.20                        19.15                      10.30           14.00       26.45         4.10     6.98
shuttle       B0-0.3    13.95                9.00         90.15                        90.15                      90.15             NaN        0.00         0.00     3.74
              SR-0.3    13.90                9.00         53.95                        20.20                      15.45            3.25       17.65         9.85    16.06
              B0-val    87.10                9.00         10.65                        10.65                      10.65             NaN        0.00         0.00     4.29
              SR-val    87.05                9.00          2.25                         1.05                       0.65            3.85        1.80         0.65    18.30
skin          B0-0.3    86.70                3.00       7858.15                      7858.15                    7858.15             NaN        0.00         0.00    21.05
              SR-0.3    86.60                3.00       1274.40                      1095.10                     756.35           34.00     1814.20       168.90   326.47
              B0-val  2012.55                3.00        399.90                       399.90                     399.90             NaN        0.00         0.00    53.18
              SR-val  2010.75                3.00        105.50                        90.30                      60.45           45.45       87.20        20.30  6401.44
magic         B0-0.3   285.60               10.00       4048.35                      4048.35                    4048.35             NaN        0.00         0.00     2.05
              SR-0.3   285.35               10.00       3235.30                      3117.50                    2714.10          191.20      488.15       295.20   801.45
              B0-val  5728.40               10.00         10.95                        10.95                      10.95             NaN        0.00         0.00    18.44
              SR-val  5728.40               10.00          0.70                         0.20                       0.10           10.80        0.25         0.25  1134.51
letter        B0-0.3   564.80               16.00       3403.00                      3403.00                    3403.00             NaN        0.00         0.00     2.03
              SR-0.3   564.45               16.00       2880.50                      2865.70                    2610.55          215.25      197.25        99.60  3647.16
              B0-val  5324.90               16.00          5.70                         5.70                       5.70             NaN        0.00         0.00     3.94
              SR-val  5324.90               16.00          0.80                         0.50                       0.05            5.50        0.15         0.25  1406.37
landsat       B0-0.3   198.85               36.00        534.25                       534.25                     534.25             NaN        0.00         0.00     0.75
              SR-0.3   198.80               36.00        389.80                       357.30                     239.50          189.00       50.05        34.35   586.46
              B0-val  2123.70               36.00          1.85                         1.85                       1.85             NaN        0.00         0.00     2.93
              SR-val  2123.70               36.00          0.45                         0.40                       0.40            1.85        0.05         0.00   154.63
page_blocks   B0-0.3    25.05                9.97        632.25                       632.25                     632.25             NaN        0.00         0.00     0.39
              SR-0.3    25.05                9.97        513.90                       510.65                     473.45           12.90       40.10        12.50    28.88
              B0-val   317.95               10.00         84.10                        84.10                      84.10             NaN        0.00         0.00     0.66
              SR-val   317.40               10.00         61.30                        47.10                      28.05           23.85        7.35         4.45    58.29
```

Gap to the better of the tuned kNN and the Random Forest (readable-baselines run, same seeds; accuracy points):

```
model          B0-0.3  SR-0.3  B0-val  SR-val
dataset                                      
S1             -10.25  -10.25   -8.73   -8.71
S2              -9.06   -9.06   -8.30   -8.30
S3             -12.19  -12.19  -11.92  -11.90
wine             0.00    0.00   -0.14   -0.14
breast_cancer   -1.40   -1.18   -0.92   -0.83
digits          -2.65   -2.65   -2.83   -2.83
thyroid         -2.24   -1.16   -2.22   -0.67
shuttle         -0.21   -0.14   -0.05   -0.04
skin            -3.97   -0.62   -0.18   -0.04
magic          -16.27  -11.20   -6.15   -6.15
letter         -21.29  -18.85   -1.32   -1.32
landsat         -6.22   -5.00   -1.30   -1.30
page_blocks    -12.19   -9.67   -1.86   -1.59
```

