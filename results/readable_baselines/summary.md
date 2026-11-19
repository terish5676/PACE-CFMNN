# Readable baselines and theta chosen by validation

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). The paper's 80/20 split, seeds 70–89 (20). Test accuracy (%), mean over seeds. Each comparison: paired two-sided Wilcoxon over seeds, Holm across datasets; better / level / worse for the first model.

## Accuracy

| Dataset | CFMNN-0.3 | CFMNN-val | Tree | LogReg | EBM | GFMM-0.3 | kNN | Random Forest |
|---|---|---|---|---|---|---|---|---|
| S1 | 89.1 | 90.6 | 97.6 | 100.0 | 100.0 | 89.2 | 96.8 | 99.4 |
| S2 | 80.8 | 81.6 | 88.0 | 90.9 | 90.5 | 79.8 | 87.8 | 89.9 |
| S3 | 87.8 | 88.1 | 99.9 | 82.1 | 99.9 | 83.4 | 89.1 | 100.0 |
| wine | 98.3 | 98.2 | 89.6 | 98.2 | 98.1 | 96.0 | 96.5 | 98.3 |
| breast_cancer | 95.4 | 95.9 | 93.7 | 97.7 | 98.3 | 95.1 | 96.8 | 96.5 |
| digits | 96.0 | 95.8 | 84.5 | 97.2 | 97.3 | 98.3 | 98.7 | 97.7 |
| thyroid | 97.4 | 97.4 | 99.6 | 95.2 | 99.6 | 96.2 | 93.9 | 99.6 |
| shuttle | 99.8 | 99.9 | 100.0 | 95.8 | 100.0 | 88.5 | 99.9 | 100.0 |
| skin | 96.0 | 99.8 | 99.9 | 91.9 | 97.2 | 98.8 | 100.0 | 100.0 |
| magic | 71.6 | 81.8 | 85.0 | 79.0 | 85.8 | 73.4 | 83.4 | 87.9 |
| letter | 75.1 | 95.1 | 86.6 | 77.3 | 87.7 | 84.1 | 95.7 | 96.4 |
| landsat | 85.1 | 90.0 | 86.6 | 85.8 | 90.0 | 86.3 | 90.6 | 91.3 |
| page_blocks | 85.0 | 95.4 | 96.8 | 95.6 | 97.3 | 90.2 | 95.8 | 97.2 |

## Primary: CFMNN-val against the readable models, and CFMNN against GFMM at the same theta (accuracy)

**CFMNN-val vs Tree** (accuracy): better 5, level 0, worse 8

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -6.97 | 0.0001 | 0.0009 | worse |
| S2 | -6.40 | 0.0000 | 0.0000 | worse |
| S3 | -11.87 | 0.0001 | 0.0009 | worse |
| wine | +8.61 | 0.0001 | 0.0009 | better |
| breast_cancer | +2.24 | 0.0008 | 0.0009 | better |
| digits | +11.38 | 0.0001 | 0.0009 | better |
| thyroid | -2.22 | 0.0001 | 0.0009 | worse |
| shuttle | -0.03 | 0.0002 | 0.0009 | worse |
| skin | -0.15 | 0.0000 | 0.0000 | worse |
| magic | -3.20 | 0.0000 | 0.0000 | worse |
| letter | +8.49 | 0.0001 | 0.0009 | better |
| landsat | +3.48 | 0.0001 | 0.0009 | better |
| page_blocks | -1.46 | 0.0001 | 0.0009 | worse |

**CFMNN-val vs LogReg** (accuracy): better 7, level 2, worse 4

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -9.36 | 0.0001 | 0.0010 | worse |
| S2 | -9.29 | 0.0001 | 0.0010 | worse |
| S3 | +6.01 | 0.0001 | 0.0010 | better |
| wine | +0.00 | 0.8608 | 0.8608 | level |
| breast_cancer | -1.75 | 0.0039 | 0.0117 | worse |
| digits | -1.35 | 0.0002 | 0.0010 | worse |
| thyroid | +2.22 | 0.0001 | 0.0010 | better |
| shuttle | +4.10 | 0.0001 | 0.0010 | better |
| skin | +7.89 | 0.0000 | 0.0000 | better |
| magic | +2.78 | 0.0001 | 0.0010 | better |
| letter | +17.77 | 0.0000 | 0.0000 | better |
| landsat | +4.24 | 0.0001 | 0.0010 | better |
| page_blocks | -0.19 | 0.0531 | 0.1063 | level |

**CFMNN-val vs EBM** (accuracy): better 2, level 2, worse 9

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -9.35 | 0.0001 | 0.0011 | worse |
| S2 | -8.89 | 0.0001 | 0.0011 | worse |
| S3 | -11.85 | 0.0001 | 0.0011 | worse |
| wine | +0.14 | 0.8514 | 1.0000 | level |
| breast_cancer | -2.37 | 0.0004 | 0.0017 | worse |
| digits | -1.49 | 0.0001 | 0.0011 | worse |
| thyroid | -2.24 | 0.0001 | 0.0011 | worse |
| shuttle | -0.03 | 0.0026 | 0.0079 | worse |
| skin | +2.54 | 0.0001 | 0.0011 | better |
| magic | -4.07 | 0.0001 | 0.0011 | worse |
| letter | +7.37 | 0.0001 | 0.0011 | better |
| landsat | +0.08 | 0.6012 | 1.0000 | level |
| page_blocks | -1.92 | 0.0001 | 0.0011 | worse |

**CFMNN-0.3 vs GFMM-0.3** (accuracy): better 4, level 3, worse 6

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -0.12 | 0.8617 | 1.0000 | level |
| S2 | +0.98 | 0.0303 | 0.0909 | level |
| S3 | +4.38 | 0.0001 | 0.0009 | better |
| wine | +2.36 | 0.0099 | 0.0471 | better |
| breast_cancer | +0.35 | 0.6157 | 1.0000 | level |
| digits | -2.33 | 0.0001 | 0.0009 | worse |
| thyroid | +1.14 | 0.0000 | 0.0000 | better |
| shuttle | +11.31 | 0.0002 | 0.0015 | better |
| skin | -2.85 | 0.0000 | 0.0000 | worse |
| magic | -1.77 | 0.0068 | 0.0408 | worse |
| letter | -8.97 | 0.0000 | 0.0000 | worse |
| landsat | -1.18 | 0.0094 | 0.0471 | worse |
| page_blocks | -5.18 | 0.0009 | 0.0062 | worse |


## Secondary: the explained model, CFMNN-0.3, against the readable models (accuracy)

**CFMNN-0.3 vs Tree** (accuracy): better 3, level 0, worse 10

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -8.48 | 0.0001 | 0.0009 | worse |
| S2 | -7.16 | 0.0001 | 0.0009 | worse |
| S3 | -12.13 | 0.0001 | 0.0009 | worse |
| wine | +8.75 | 0.0003 | 0.0009 | better |
| breast_cancer | +1.75 | 0.0036 | 0.0036 | better |
| digits | +11.56 | 0.0001 | 0.0009 | better |
| thyroid | -2.23 | 0.0001 | 0.0009 | worse |
| shuttle | -0.20 | 0.0001 | 0.0009 | worse |
| skin | -3.94 | 0.0000 | 0.0000 | worse |
| magic | -13.32 | 0.0000 | 0.0000 | worse |
| letter | -11.48 | 0.0001 | 0.0009 | worse |
| landsat | -1.43 | 0.0010 | 0.0020 | worse |
| page_blocks | -11.79 | 0.0000 | 0.0000 | worse |

**CFMNN-0.3 vs LogReg** (accuracy): better 4, level 2, worse 7

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -10.88 | 0.0001 | 0.0009 | worse |
| S2 | -10.04 | 0.0001 | 0.0009 | worse |
| S3 | +5.74 | 0.0001 | 0.0009 | better |
| wine | +0.14 | 0.4685 | 0.4685 | level |
| breast_cancer | -2.24 | 0.0028 | 0.0083 | worse |
| digits | -1.17 | 0.0011 | 0.0045 | worse |
| thyroid | +2.20 | 0.0001 | 0.0009 | better |
| shuttle | +3.94 | 0.0000 | 0.0000 | better |
| skin | +4.10 | 0.0000 | 0.0000 | better |
| magic | -7.34 | 0.0001 | 0.0009 | worse |
| letter | -2.20 | 0.0001 | 0.0009 | worse |
| landsat | -0.68 | 0.0289 | 0.0579 | level |
| page_blocks | -10.52 | 0.0000 | 0.0000 | worse |

**CFMNN-0.3 vs EBM** (accuracy): better 0, level 1, worse 12

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -10.87 | 0.0001 | 0.0010 | worse |
| S2 | -9.65 | 0.0000 | 0.0000 | worse |
| S3 | -12.12 | 0.0001 | 0.0010 | worse |
| wine | +0.28 | 0.4927 | 0.4927 | level |
| breast_cancer | -2.85 | 0.0001 | 0.0010 | worse |
| digits | -1.31 | 0.0003 | 0.0010 | worse |
| thyroid | -2.26 | 0.0001 | 0.0010 | worse |
| shuttle | -0.19 | 0.0001 | 0.0010 | worse |
| skin | -1.25 | 0.0003 | 0.0010 | worse |
| magic | -14.20 | 0.0001 | 0.0010 | worse |
| letter | -12.60 | 0.0001 | 0.0010 | worse |
| landsat | -4.84 | 0.0001 | 0.0010 | worse |
| page_blocks | -12.25 | 0.0000 | 0.0000 | worse |


## Secondary: theta by validation against theta = 0.3 (accuracy)

**CFMNN-val vs CFMNN-0.3** (accuracy): better 7, level 6, worse 0

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | +1.51 | 0.0005 | 0.0035 | better |
| S2 | +0.76 | 0.0825 | 0.4951 | level |
| S3 | +0.26 | 0.6579 | 1.0000 | level |
| wine | -0.14 | 0.5099 | 1.0000 | level |
| breast_cancer | +0.48 | 0.2201 | 1.0000 | level |
| digits | -0.18 | 0.5828 | 1.0000 | level |
| thyroid | +0.02 | 0.8684 | 1.0000 | level |
| shuttle | +0.16 | 0.0001 | 0.0008 | better |
| skin | +3.79 | 0.0000 | 0.0000 | better |
| magic | +10.12 | 0.0000 | 0.0000 | better |
| letter | +19.97 | 0.0000 | 0.0000 | better |
| landsat | +4.91 | 0.0001 | 0.0008 | better |
| page_blocks | +10.33 | 0.0000 | 0.0000 | better |


## Secondary: the primary comparisons by macro-F1

**CFMNN-val vs Tree** (macro_f1): better 5, level 0, worse 8

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -6.97 | 0.0000 | 0.0000 | worse |
| S2 | -6.41 | 0.0000 | 0.0000 | worse |
| S3 | -12.95 | 0.0000 | 0.0000 | worse |
| wine | +8.62 | 0.0001 | 0.0005 | better |
| breast_cancer | +2.41 | 0.0003 | 0.0008 | better |
| digits | +11.38 | 0.0000 | 0.0000 | better |
| thyroid | -10.96 | 0.0000 | 0.0000 | worse |
| shuttle | -3.18 | 0.0240 | 0.0240 | worse |
| skin | -0.23 | 0.0000 | 0.0000 | worse |
| magic | -3.36 | 0.0000 | 0.0000 | worse |
| letter | +8.48 | 0.0000 | 0.0000 | better |
| landsat | +4.73 | 0.0000 | 0.0000 | better |
| page_blocks | -5.16 | 0.0005 | 0.0010 | worse |

**CFMNN-val vs LogReg** (macro_f1): better 7, level 2, worse 4

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -9.37 | 0.0001 | 0.0004 | worse |
| S2 | -9.29 | 0.0000 | 0.0000 | worse |
| S3 | +7.30 | 0.0000 | 0.0000 | better |
| wine | +0.01 | 0.6725 | 0.6725 | level |
| breast_cancer | -1.86 | 0.0079 | 0.0237 | worse |
| digits | -1.37 | 0.0000 | 0.0000 | worse |
| thyroid | +14.92 | 0.0000 | 0.0000 | better |
| shuttle | +51.35 | 0.0000 | 0.0000 | better |
| skin | +11.82 | 0.0000 | 0.0000 | better |
| magic | +4.02 | 0.0000 | 0.0000 | better |
| letter | +17.99 | 0.0000 | 0.0000 | better |
| landsat | +6.73 | 0.0000 | 0.0000 | better |
| page_blocks | +2.02 | 0.0266 | 0.0533 | level |

**CFMNN-val vs EBM** (macro_f1): better 2, level 3, worse 8

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -9.36 | 0.0001 | 0.0004 | worse |
| S2 | -8.90 | 0.0000 | 0.0000 | worse |
| S3 | -12.92 | 0.0000 | 0.0000 | worse |
| wine | +0.15 | 0.8582 | 0.8582 | level |
| breast_cancer | -2.54 | 0.0004 | 0.0014 | worse |
| digits | -1.50 | 0.0000 | 0.0000 | worse |
| thyroid | -11.04 | 0.0000 | 0.0000 | worse |
| shuttle | +2.11 | 0.2774 | 0.5547 | level |
| skin | +3.79 | 0.0000 | 0.0000 | better |
| magic | -4.36 | 0.0000 | 0.0000 | worse |
| letter | +7.40 | 0.0000 | 0.0000 | better |
| landsat | +0.54 | 0.0266 | 0.0799 | level |
| page_blocks | -8.68 | 0.0000 | 0.0001 | worse |

**CFMNN-0.3 vs GFMM-0.3** (macro_f1): better 4, level 5, worse 4

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| S1 | -0.12 | 0.8695 | 1.0000 | level |
| S2 | +0.98 | 0.0328 | 0.1638 | level |
| S3 | +5.30 | 0.0000 | 0.0000 | better |
| wine | +2.33 | 0.0069 | 0.0416 | better |
| breast_cancer | +0.36 | 0.6477 | 1.0000 | level |
| digits | -2.35 | 0.0000 | 0.0000 | worse |
| thyroid | +7.15 | 0.0000 | 0.0000 | better |
| shuttle | +7.28 | 0.0049 | 0.0340 | better |
| skin | -4.77 | 0.0000 | 0.0000 | worse |
| magic | -1.02 | 0.0973 | 0.3892 | level |
| letter | -9.07 | 0.0000 | 0.0000 | worse |
| landsat | -1.58 | 0.0006 | 0.0047 | worse |
| page_blocks | -0.90 | 0.9563 | 1.0000 | level |


## Descriptive

Gap to the better of the tuned kNN and the Random Forest (points): CFMNN-0.3 -21.3 to +0.0, CFMNN-val -11.9 to -0.0

Settings chosen (count over seeds):

- CFMNN-val, letter: 0.05 ×20
- Tree, letter: 1024.0 ×20
- LogReg, letter: 100.0 ×20
- kNN, letter: 1.0 ×20
- CFMNN-val, skin: 0.05 ×20
- Tree, skin: 256.0 ×20
- LogReg, skin: 100.0 ×15, 10.0 ×5
- kNN, skin: 1.0 ×13, 3.0 ×7
- CFMNN-val, shuttle: 0.05 ×20
- Tree, shuttle: 32.0 ×16, 64.0 ×4
- LogReg, shuttle: 100.0 ×20
- kNN, shuttle: 1.0 ×20
- CFMNN-val, S1: 0.5 ×17, 0.4 ×3
- Tree, S1: 64.0 ×11, 32.0 ×7, 128.0 ×2
- LogReg, S1: 0.1 ×12, 1.0 ×8
- kNN, S1: 20.0 ×20
- CFMNN-val, S2: 0.5 ×11, 0.4 ×9
- Tree, S2: 32.0 ×16, 64.0 ×3, 16.0 ×1
- LogReg, S2: 1.0 ×9, 10.0 ×5, 100.0 ×4, 0.1 ×2
- kNN, S2: 20.0 ×20
- CFMNN-val, S3: 0.4 ×13, 0.5 ×6, 0.3 ×1
- Tree, S3: 4.0 ×13, 8.0 ×7
- LogReg, S3: 10.0 ×7, 0.1 ×6, 1.0 ×6, 100.0 ×1
- kNN, S3: 20.0 ×10, 5.0 ×9, 10.0 ×1
- CFMNN-val, magic: 0.05 ×20
- Tree, magic: 128.0 ×13, 64.0 ×6, 256.0 ×1
- LogReg, magic: 10.0 ×9, 100.0 ×8, 1.0 ×3
- kNN, magic: 5.0 ×11, 10.0 ×7, 20.0 ×2
- CFMNN-val, landsat: 0.05 ×13, 0.1 ×7
- Tree, landsat: 64.0 ×10, 128.0 ×9, 32.0 ×1
- LogReg, landsat: 100.0 ×18, 10.0 ×2
- kNN, landsat: 3.0 ×12, 5.0 ×8
- CFMNN-val, wine: 0.2 ×6, 0.4 ×4, 0.3 ×4, 0.5 ×3, 0.15 ×2, 0.05 ×1
- Tree, wine: 8.0 ×13, 4.0 ×4, 16.0 ×3
- LogReg, wine: 1.0 ×14, 10.0 ×3, 0.1 ×2, 100.0 ×1
- kNN, wine: 20.0 ×9, 10.0 ×8, 5.0 ×2, 3.0 ×1
- CFMNN-val, breast_cancer: 0.15 ×7, 0.2 ×5, 0.1 ×4, 0.05 ×2, 0.3 ×2
- Tree, breast_cancer: 8.0 ×9, 4.0 ×7, 16.0 ×4
- LogReg, breast_cancer: 10.0 ×12, 100.0 ×8
- kNN, breast_cancer: 10.0 ×8, 3.0 ×5, 5.0 ×5, 1.0 ×2
- CFMNN-val, digits: 0.2 ×8, 0.3 ×8, 0.15 ×4
- Tree, digits: 128.0 ×16, 64.0 ×4
- LogReg, digits: 10.0 ×14, 1.0 ×6
- kNN, digits: 1.0 ×14, 3.0 ×5, 5.0 ×1
- CFMNN-val, thyroid: 0.5 ×11, 0.4 ×7, 0.3 ×2
- Tree, thyroid: 16.0 ×9, 8.0 ×8, 32.0 ×3
- LogReg, thyroid: 100.0 ×20
- kNN, thyroid: 5.0 ×14, 10.0 ×5, 3.0 ×1
- CFMNN-val, page_blocks: 0.05 ×19, 0.1 ×1
- Tree, page_blocks: 16.0 ×11, 32.0 ×9
- LogReg, page_blocks: 100.0 ×20
- kNN, page_blocks: 3.0 ×10, 1.0 ×9, 5.0 ×1

Mean size and time per dataset (rules; conditions per rule; fit s; predict s):

                               rules  conditions_per_rule    fit_s  predict_s
dataset       model                                                          
S1            CFMNN-0.3      1516.20                10.00     1.06       0.60
              CFMNN-val       788.50                10.00     0.81       0.29
              EBM              10.00                 1.00   249.66       0.00
              GFMM-0.3       2365.55                10.00   164.72       1.26
              LogReg           10.00                  NaN     0.01       0.00
              Random Forest      NaN                  NaN     2.89       0.07
              Tree             53.60                 3.28     0.04       0.00
              kNN                NaN                  NaN     0.01       0.21
S2            CFMNN-0.3      1533.30                10.00     1.11       0.62
              CFMNN-val       924.55                10.00     0.92       0.38
              EBM              10.00                 1.00    20.64       0.00
              GFMM-0.3       2429.90                10.00   168.46       1.29
              LogReg           10.00                  NaN     0.01       0.00
              Random Forest      NaN                  NaN     3.31       0.07
              Tree             36.00                 3.11     0.04       0.00
              kNN                NaN                  NaN     0.01       0.21
S3            CFMNN-0.3      1525.35                10.00     1.14       0.61
              CFMNN-val      1032.15                10.00     0.96       0.39
              EBM              10.00                 1.00    46.02       0.00
              GFMM-0.3       2415.75                10.00   160.98       1.27
              LogReg           10.00                  NaN     0.01       0.00
              Random Forest      NaN                  NaN     2.08       0.06
              Tree              4.40                 2.25     0.02       0.00
              kNN                NaN                  NaN     0.01       0.17
breast_cancer CFMNN-0.3        42.75                29.99     0.06       0.01
              CFMNN-val       140.35                30.00     0.07       0.01
              EBM              30.00                 1.00    34.15       0.00
              GFMM-0.3         98.85                30.00     1.33       0.03
              LogReg           30.00                  NaN     0.01       0.00
              Random Forest      NaN                  NaN     0.80       0.04
              Tree              8.20                 2.90     0.01       0.00
              kNN                NaN                  NaN     0.00       0.00
digits        CFMNN-0.3       294.35                62.42     0.23       0.25
              CFMNN-val       431.65                63.20     0.26       0.41
              EBM              64.00                 1.00    60.80       0.00
              GFMM-0.3       1399.55                64.00     6.12       2.61
              LogReg          607.00                  NaN     0.15       0.00
              Random Forest      NaN                  NaN     1.39       0.06
              Tree            115.20                 8.02     0.04       0.00
              kNN                NaN                  NaN     0.00       0.01
landsat       CFMNN-0.3       198.85                36.00     0.62       0.30
              CFMNN-val      2123.70                36.00     1.68       3.57
              EBM              36.00                 1.00   118.70       0.01
              GFMM-0.3        266.60                36.00    47.47       0.64
              LogReg          216.00                  NaN     1.10       0.00
              Random Forest      NaN                  NaN     4.34       0.09
              Tree             91.20                 6.31     0.11       0.00
              kNN                NaN                  NaN     0.00       0.09
letter        CFMNN-0.3       564.80                16.00     2.08       2.96
              CFMNN-val      5324.90                16.00     3.56      38.03
              EBM              16.00                 1.00  3107.46       0.01
              GFMM-0.3        741.90                16.00   601.64       2.61
              LogReg          416.00                  NaN     7.89       0.00
              Random Forest      NaN                  NaN     8.43       0.34
              Tree           1024.00                 9.29     0.21       0.00
              kNN                NaN                  NaN     0.00       0.65
magic         CFMNN-0.3       285.60                10.00     1.74       0.48
              CFMNN-val      5728.40                10.00    10.30       8.25
              EBM              10.00                 1.00   115.88       0.01
              GFMM-0.3        320.85                10.00   119.18       1.20
              LogReg           10.00                  NaN     0.06       0.00
              Random Forest      NaN                  NaN    18.99       0.22
              Tree            115.20                 5.26     0.28       0.00
              kNN                NaN                  NaN     0.05       0.35
page_blocks   CFMNN-0.3        25.05                 9.97     0.37       0.01
              CFMNN-val       317.95                10.00     0.61       0.16
              EBM              10.00                 1.00    33.83       0.00
              GFMM-0.3         84.05                10.00     6.71       0.19
              LogReg           50.00                  NaN     0.25       0.00
              Random Forest      NaN                  NaN     3.06       0.06
              Tree             23.20                 4.03     0.04       0.00
              kNN                NaN                  NaN     0.01       0.02
shuttle       CFMNN-0.3        13.95                 9.00     3.33       0.09
              CFMNN-val        87.10                 9.00     3.81       0.49
              EBM               9.00                 1.00  2597.52       0.01
              GFMM-0.3         33.30                 9.00    49.24       1.82
              LogReg           63.00                  NaN     3.19       0.00
              Random Forest      NaN                  NaN    10.38       0.24
              Tree             32.00                 3.82     0.20       0.00
              kNN                NaN                  NaN     0.21       0.57
skin          CFMNN-0.3        86.70                 3.00    17.86       1.21
              CFMNN-val      2012.55                 3.00    40.23      23.30
              EBM               3.00                 1.00  2779.91       0.01
              GFMM-0.3         68.60                 3.00    56.97       8.69
              LogReg            3.00                  NaN     0.43       0.00
              Random Forest      NaN                  NaN    45.53       0.78
              Tree            256.00                 2.94     0.65       0.01
              kNN                NaN                  NaN     0.54       0.91
thyroid       CFMNN-0.3        53.85                16.62     0.56       0.05
              CFMNN-val        21.95                14.71     0.50       0.02
              EBM              21.00                 1.00    41.41       0.00
              GFMM-0.3        360.80                21.00    14.89       0.62
              LogReg           62.55                  NaN     0.22       0.00
              Random Forest      NaN                  NaN     1.34       0.06
              Tree             13.60                 4.47     0.01       0.00
              kNN                NaN                  NaN     0.00       0.08
wine          CFMNN-0.3        31.35                13.00     0.02       0.00
              CFMNN-val        41.10                12.99     0.02       0.00
              EBM              13.00                 1.00     8.21       0.00
              GFMM-0.3         65.25                13.00     0.29       0.01
              LogReg           39.00                  NaN     0.01       0.00
              Random Forest      NaN                  NaN     0.62       0.04
              Tree              7.10                 2.85     0.00       0.00
              kNN                NaN                  NaN     0.00       0.00

