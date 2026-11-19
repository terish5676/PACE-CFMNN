# Run H: the order of training errors in self-repair

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Run A's split and SR arm, seeds 130–139 (10), cached gate. Each comparison: paired two-sided Wilcoxon over seeds, Holm across datasets; better / level / worse for the first arm. Written by `scripts/run_repair_order.py`.

## Primary: greedy-300 vs random-300, test macro-F1

**greedy-300 vs random-300**: better 2, level 3, worse 1

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | +0.22 | 0.2500 | 0.5000 | level |
| skin | -1.08 | 0.0059 | 0.0234 | worse |
| magic | +4.87 | 0.0020 | 0.0117 | better |
| letter | +2.98 | 0.0020 | 0.0117 | better |
| landsat | +0.43 | 0.0840 | 0.2520 | level |
| page_blocks | +0.29 | 0.3750 | 0.5000 | level |

## Secondary comparisons, test macro-F1

**box-300 vs random-300**: better 0, level 4, worse 2

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | -0.75 | 0.7500 | 1.0000 | level |
| skin | -2.57 | 0.0020 | 0.0117 | worse |
| magic | -0.60 | 0.1309 | 0.5234 | level |
| letter | -0.64 | 0.0059 | 0.0293 | worse |
| landsat | +0.01 | 0.9219 | 1.0000 | level |
| page_blocks | -0.47 | 0.9219 | 1.0000 | level |

**greedy-300 vs random-all**: better 1, level 5, worse 0

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | +0.22 | 0.2500 | 0.7500 | level |
| skin | -0.16 | 0.6250 | 1.0000 | level |
| magic | +7.97 | 0.0020 | 0.0117 | better |
| letter | +0.79 | 0.0137 | 0.0684 | level |
| landsat | +0.44 | 0.0840 | 0.3359 | level |
| page_blocks | +0.14 | 0.5703 | 1.0000 | level |

**greedy-1000 vs random-1000**: better 2, level 4, worse 0

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | +0.22 | 0.2500 | 0.5000 | level |
| skin | -0.98 | 0.0371 | 0.1484 | level |
| magic | +2.53 | 0.0059 | 0.0293 | better |
| letter | +1.58 | 0.0020 | 0.0117 | better |
| landsat | +0.42 | 0.0840 | 0.2520 | level |
| page_blocks | +0.14 | 0.5703 | 0.5703 | level |

**box-1000 vs random-1000**: better 0, level 6, worse 0

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | -0.75 | 0.7500 | 1.0000 | level |
| skin | -2.34 | 0.0098 | 0.0586 | level |
| magic | +1.43 | 0.0195 | 0.0977 | level |
| letter | -0.46 | 0.1934 | 0.7734 | level |
| landsat | +0.04 | 0.8457 | 1.0000 | level |
| page_blocks | -0.66 | 0.4922 | 1.0000 | level |

**greedy-300 vs box-300**: better 2, level 4, worse 0

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | +0.97 | 0.1250 | 0.3164 | level |
| skin | +1.49 | 0.0645 | 0.2578 | level |
| magic | +5.47 | 0.0020 | 0.0117 | better |
| letter | +3.62 | 0.0020 | 0.0117 | better |
| landsat | +0.41 | 0.1055 | 0.3164 | level |
| page_blocks | +0.75 | 0.3594 | 0.3594 | level |

## Amendment 1: greedy-300 vs random-300-real, test macro-F1

**greedy-300 vs random-300-real**: better 2, level 4, worse 0

| Dataset | Mean difference (points) | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | +0.22 | 0.2500 | 0.7500 | level |
| skin | -0.16 | 0.6250 | 1.0000 | level |
| magic | +4.65 | 0.0020 | 0.0117 | better |
| letter | +2.67 | 0.0020 | 0.0117 | better |
| landsat | +0.44 | 0.0840 | 0.3359 | level |
| page_blocks | +0.13 | 0.5703 | 1.0000 | level |

## Amendment 1: how to read the primary verdict

| Dataset | Primary | Against random-300-real | Reading |
|---|---|---|---|
| thyroid | level | level | level (against real tries: level) |
| skin | worse | level | worse (against real tries: level) |
| magic | better | better | a better order |
| letter | better | better | a better order |
| landsat | level | level | level (against real tries: level) |
| page_blocks | level | level | level (against real tries: level) |

## Amendment 1, efficiency: greedy-300 vs random-all, seconds

**greedy-300 vs random-all** (seconds): slower 6

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | +1.5 | 0.0020 | 0.0117 | slower |
| skin | +1133.3 | 0.0020 | 0.0117 | slower |
| magic | +224.3 | 0.0020 | 0.0117 | slower |
| letter | +376.9 | 0.0020 | 0.0117 | slower |
| landsat | +15.8 | 0.0020 | 0.0117 | slower |
| page_blocks | +11.8 | 0.0020 | 0.0117 | slower |

## Amendment 1, efficiency: greedy-300 vs random-all, edits_gated

**greedy-300 vs random-all** (edits_gated): level 4, fewer 2

| Dataset | Mean difference | p | Holm p | Verdict |
|---|---|---|---|---|
| thyroid | -3.6 | 0.0156 | 0.0625 | level |
| skin | -3.6 | 0.7695 | 1.0000 | level |
| magic | -525.6 | 0.0020 | 0.0117 | fewer |
| letter | -961.3 | 0.0020 | 0.0117 | fewer |
| landsat | -0.2 | 0.7695 | 1.0000 | level |
| page_blocks | -2.0 | 0.6973 | 1.0000 | level |

## Test macro-F1 (%), mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 84.29 | 91.29 | 91.29 | 91.29 | 91.29 | 90.54 | 90.54 | 91.51 | 91.51 |
| landsat | 82.78 | 84.23 | 84.22 | 84.22 | 84.22 | 84.25 | 84.26 | 84.66 | 84.64 |
| page_blocks | 62.57 | 65.50 | 65.64 | 65.64 | 65.65 | 65.03 | 64.99 | 65.79 | 65.79 |
| magic | 71.30 | 74.67 | 73.51 | 71.57 | 74.88 | 74.07 | 74.94 | 79.54 | 76.03 |
| letter | 73.75 | 76.05 | 77.44 | 78.25 | 76.36 | 75.41 | 76.98 | 79.03 | 79.02 |
| skin | 93.20 | 98.88 | 98.75 | 97.96 | 97.96 | 96.31 | 96.40 | 97.80 | 97.77 |

## Test accuracy (%), mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 96.89 | 98.19 | 98.19 | 98.19 | 98.19 | 98.06 | 98.06 | 98.25 | 98.25 |
| landsat | 85.53 | 86.52 | 86.51 | 86.51 | 86.51 | 86.60 | 86.61 | 86.94 | 86.91 |
| page_blocks | 88.65 | 90.24 | 90.46 | 90.46 | 90.47 | 90.53 | 90.42 | 91.04 | 91.04 |
| magic | 72.58 | 76.88 | 75.67 | 73.48 | 77.11 | 75.86 | 77.54 | 82.00 | 78.30 |
| letter | 73.97 | 76.20 | 77.56 | 78.37 | 76.52 | 75.61 | 77.16 | 79.15 | 79.12 |
| skin | 95.89 | 99.27 | 99.18 | 98.68 | 98.68 | 97.66 | 97.73 | 98.59 | 98.57 |

## Lowest class recall (%), mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 60.15 | 82.65 | 82.65 | 82.65 | 82.65 | 80.43 | 80.43 | 83.58 | 83.58 |
| landsat | 48.14 | 56.16 | 56.16 | 56.16 | 56.16 | 55.83 | 55.83 | 56.63 | 56.63 |
| page_blocks | 34.24 | 39.85 | 39.85 | 39.85 | 39.85 | 36.36 | 37.88 | 38.18 | 38.18 |
| magic | 70.50 | 67.45 | 67.10 | 67.78 | 67.31 | 70.12 | 64.50 | 67.36 | 67.69 |
| letter | 47.74 | 53.76 | 55.56 | 56.02 | 54.02 | 50.70 | 53.66 | 57.92 | 57.91 |
| skin | 81.85 | 97.86 | 97.30 | 94.74 | 94.74 | 90.56 | 90.73 | 93.95 | 93.80 |

## Training error at the end (%), mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 1.88 | 0.39 | 0.39 | 0.39 | 0.39 | 0.45 | 0.45 | 0.39 | 0.39 |
| landsat | 8.63 | 6.12 | 6.13 | 6.13 | 6.13 | 6.27 | 6.23 | 5.68 | 5.74 |
| page_blocks | 10.65 | 8.41 | 8.30 | 8.30 | 8.31 | 8.36 | 8.34 | 8.01 | 8.01 |
| magic | 25.20 | 20.14 | 21.19 | 23.20 | 20.09 | 21.77 | 20.10 | 12.10 | 18.59 |
| letter | 21.08 | 17.27 | 14.62 | 13.39 | 16.87 | 18.76 | 15.84 | 11.06 | 12.01 |
| skin | 4.07 | 0.71 | 0.80 | 1.30 | 1.30 | 2.33 | 2.26 | 1.41 | 1.44 |

## Edits that reached the gate, mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid |  | 24 | 24 | 24 | 24 | 24 | 24 | 20 | 20 |
| landsat |  | 187 | 188 | 188 | 188 | 181 | 183 | 188 | 189 |
| page_blocks |  | 174 | 201 | 201 | 195 | 176 | 214 | 199 | 204 |
| magic |  | 191 | 471 | 826 | 300 | 44 | 190 | 300 | 960 |
| letter |  | 254 | 698 | 1261 | 300 | 159 | 578 | 300 | 1000 |
| skin |  | 39 | 71 | 146 | 146 | 8 | 13 | 143 | 144 |

## Edits accepted, mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid |  | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 13 |
| landsat |  | 165 | 166 | 166 | 166 | 161 | 163 | 172 | 173 |
| page_blocks |  | 14 | 15 | 15 | 15 | 9 | 13 | 12 | 12 |
| magic |  | 188 | 465 | 815 | 296 | 43 | 187 | 300 | 958 |
| letter |  | 200 | 520 | 906 | 233 | 89 | 368 | 300 | 921 |
| skin |  | 34 | 57 | 102 | 102 | 8 | 13 | 111 | 110 |

## Tries skipped: already right, mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid |  | 44 | 44 | 44 | 44 | 43 | 43 | 47 | 47 |
| landsat |  | 86 | 86 | 86 | 86 | 93 | 93 | 84 | 85 |
| page_blocks |  | 71 | 90 | 90 | 90 | 70 | 78 | 46 | 87 |
| magic |  | 108 | 500 | 1374 | 241 | 256 | 806 | 0 | 36 |
| letter |  | 45 | 293 | 802 | 60 | 140 | 417 | 0 | 0 |
| skin |  | 260 | 913 | 4731 | 4731 | 292 | 987 | 152 | 835 |

## Test predictions fixed against B0, mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 0.0 | 10.6 | 10.6 | 10.6 | 10.6 | 9.8 | 9.8 | 11.0 | 11.0 |
| landsat | 0.0 | 21.8 | 21.7 | 21.7 | 21.7 | 20.7 | 20.8 | 23.1 | 22.9 |
| page_blocks | 0.0 | 19.2 | 20.6 | 20.6 | 20.6 | 15.6 | 18.0 | 17.6 | 17.6 |
| magic | 0.0 | 231.5 | 287.5 | 301.6 | 266.2 | 109.4 | 231.2 | 282.2 | 325.4 |
| letter | 0.0 | 93.8 | 145.7 | 180.1 | 102.9 | 54.7 | 127.0 | 150.0 | 189.4 |
| skin | 0.0 | 940.6 | 953.1 | 961.6 | 961.6 | 450.2 | 476.4 | 961.7 | 970.5 |

## Test predictions broken against B0, mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 0.0 | 1.2 | 1.2 | 1.2 | 1.2 | 1.4 | 1.4 | 1.2 | 1.2 |
| landsat | 0.0 | 15.4 | 15.4 | 15.4 | 15.4 | 13.8 | 13.8 | 14.0 | 14.0 |
| page_blocks | 0.0 | 10.5 | 10.7 | 10.7 | 10.6 | 5.3 | 8.3 | 4.5 | 4.5 |
| magic | 0.0 | 149.7 | 228.7 | 284.5 | 179.9 | 47.0 | 136.8 | 103.0 | 216.5 |
| letter | 0.0 | 49.1 | 73.9 | 92.1 | 52.0 | 21.9 | 63.2 | 46.4 | 86.4 |
| skin | 0.0 | 113.5 | 147.3 | 278.6 | 278.6 | 16.3 | 27.3 | 300.9 | 314.7 |

## Trial edits scored (greedy only), mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid |  | 0 | 0 | 0 | 0 | 0 | 0 | 124 | 124 |
| landsat |  | 0 | 0 | 0 | 0 | 0 | 0 | 1066 | 1062 |
| page_blocks |  | 0 | 0 | 0 | 0 | 0 | 0 | 1190 | 1094 |
| magic |  | 0 | 0 | 0 | 0 | 0 | 0 | 10898 | 8611 |
| letter |  | 0 | 0 | 0 | 0 | 0 | 0 | 13679 | 10300 |
| skin |  | 0 | 0 | 0 | 0 | 0 | 0 | 6568 | 6007 |

## Seconds, mean over seeds

| Dataset | B0 | random-300 | random-1000 | random-all | random-300-real | box-300 | box-1000 | greedy-300 | greedy-1000 |
|---|---|---|---|---|---|---|---|---|---|
| thyroid | 0.0 | 0.4 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 2.0 | 1.9 |
| landsat | 0.0 | 3.3 | 3.3 | 3.3 | 3.3 | 3.2 | 3.3 | 19.1 | 18.9 |
| page_blocks | 0.0 | 1.9 | 2.2 | 2.2 | 2.1 | 1.9 | 2.3 | 14.0 | 11.6 |
| magic | 0.0 | 5.2 | 10.5 | 16.8 | 7.4 | 2.4 | 5.5 | 241.2 | 181.0 |
| letter | 0.0 | 11.8 | 25.5 | 42.8 | 13.0 | 9.4 | 22.7 | 419.7 | 348.0 |
| skin | 0.0 | 8.8 | 14.2 | 26.6 | 26.2 | 4.4 | 5.2 | 1159.9 | 994.2 |

## Training errors at the start (mean over seeds)

| thyroid | landsat | page_blocks | magic | letter | skin |
|---|---|---|---|---|---|
| 68 | 278 | 291 | 2396 | 2108 | 4992 |

