# Run L: Covertype with theta chosen by validation, then self-repair

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Descriptive: no comparison is tested. Run K's split, seeds 210–214 (5), γ = 4; 46,480 training rows, 11,621 test rows. Mean ± standard deviation over seeds. Written by `scripts/run_covertype_theta.py`.

**Not the full data:** share of the rows used = [0.1].

## θ chosen by validation

| Seed | θ | Rows used for tuning | Seconds to tune |
|---|---|---|---|
| 210 | 0.05 | 46,480 | 278 |
| 211 | 0.05 | 46,480 | 265 |
| 212 | 0.05 | 46,480 | 163 |
| 213 | 0.05 | 46,480 | 206 |
| 214 | 0.05 | 46,480 | 158 |

## The four models

| Model | Test accuracy (%) | Test macro-F1 (%) | Boxes | Training error (%) | Training errors that are ties (%) | Test predictions decided by the nearest centroid (%) | Edits accepted | Seconds (fit or repair) | PACE-flip ms | PACE-SHAP ms |
|---|---|---|---|---|---|---|---|---|---|---|
| B0-0.3 | 50.73 ± 1.54 | 39.53 ± 2.94 | 82.80 ± 7.79 | 48.86 ± 1.58 | 100.00 | 97.30 | – | 0.87 ± 0.05 | 0.84 ± 0.05 | 298.85 ± 33.43 |
| SR-0.3 | 53.86 ± 1.93 | 41.41 ± 3.93 | 78.00 ± 7.65 | 45.61 ± 1.86 | 96.82 | 93.76 | 25.80 ± 9.58 | 25.10 ± 1.89 | 0.69 ± 0.05 | 264.07 ± 25.53 |
| B0-val | 72.93 ± 0.46 | 64.27 ± 1.44 | 3,009.60 ± 143.97 | 19.60 ± 0.32 | 100.00 | 41.01 | – | 8.82 ± 0.46 | 8.66 ± 0.35 | 9,253.93 ± 407.73 |
| SR-val | 73.37 ± 0.57 | 64.82 ± 1.37 | 3,009.60 ± 143.97 | 18.90 ± 0.27 | 95.59 | 39.33 | 281.40 ± 5.73 | 447.77 ± 18.59 | 8.94 ± 0.13 | 9,241.73 ± 387.64 |

Ties: training errors inside a box of both classes (Proposition 1's form), pooled over seeds. Centroid-decided: pooled over seeds.

## For reference: Run K (seeds 170–174, not rerun)

| Model | Test accuracy (%) | Test macro-F1 (%) |
|---|---|---|
| CFMNN (Run K) | 51.45 ± 2.75 | 39.11 ± 1.47 |
| 10-NN (Run K) | 92.67 ± 0.08 | 87.11 ± 0.25 |
| Random Forest (Run K) | 95.63 ± 0.05 | 92.69 ± 0.21 |

## Cost of the run

Minutes per seed: 38.12 ± 2.32. Peak memory of the process (MB): 3,997.89 ± 0.00. Gate rows: [46480].

