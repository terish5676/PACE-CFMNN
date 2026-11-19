# Counterfactuals: PACE vs DiCE, tested over seeds

From B1 (`extensions_v1/counterfactuals/runs.csv`, seeds 40–49, 30 test points per seed), on the points where both methods found a valid counterfactual. Each cell is PACE / DiCE: the mean over seeds of each seed's mean (for time, each seed's median per counterfactual). Several runs shared the CPU, so compare times within a row only. Bold: significant over seeds (two-sided Wilcoxon, Holm across the 11 datasets, p < 0.05). Post hoc. Made by `scripts/counterfactual_table_seeds.py`.

| Dataset | Size of change | Features changed | Distance to real data of the target class | Time (ms) |
|---|---|---|---|---|
| S1 | 0.39 / 0.41 | **1.58 / 1.32** | **1.16 / 1.23** | **7.0 / 6,784** |
| S2 | 0.36 / 0.39 | **1.56 / 1.32** | **1.11 / 1.19** | **7.2 / 6,996** |
| S3 | 0.37 / 0.39 | 1.44 / 1.36 | **1.14 / 1.24** | **6.4 / 6,700** |
| Wine | **0.70 / 0.80** | **3.44 / 2.35** | **1.32 / 1.67** | **4.0 / 560** |
| Breast Cancer | 1.43 / 1.66 | **10.96 / 4.51** | **1.68 / 3.01** | **9.5 / 1,243** |
| Thyroid | **0.16 / 0.29** | **1.90 / 1.60** | **0.25 / 0.51** | **3.0 / 961** |
| Shuttle | **0.12 / 0.25** | **3.29 / 1.37** | **0.15 / 0.28** | **7.6 / 1,045** |
| MAGIC | **0.25 / 0.33** | **2.56 / 1.41** | **0.26 / 0.46** | **7.8 / 2,966** |
| Letter | **0.21 / 0.32** | **2.66 / 1.45** | **0.72 / 0.92** | **30.8 / 10,622** |
| Landsat | 1.02 / 1.10 | **10.64 / 3.32** | **1.69 / 2.53** | **24.4 / 3,767** |
| Page Blocks | 0.24 / 0.35 | **2.18 / 1.37** | **0.24 / 0.51** | **7.9 / 555** |

- Size of change: PACE better on 5 of 11 (5 of the 8 real datasets), level on 6, worse on 0. Better on: Wine, Thyroid, Shuttle, MAGIC, Letter.
- Features changed: PACE better on 0 of 11 (0 of the 8 real datasets), level on 1, worse on 10. Better on: none.
- Distance to real data of the target class: PACE better on 11 of 11 (8 of the 8 real datasets), level on 0, worse on 0. Better on: S1, S2, S3, Wine, Breast Cancer, Thyroid, Shuttle, MAGIC, Letter, Landsat, Page Blocks.
- Time (ms): PACE better on 11 of 11 (8 of the 8 real datasets), level on 0, worse on 0. Better on: S1, S2, S3, Wine, Breast Cancer, Thyroid, Shuttle, MAGIC, Letter, Landsat, Page Blocks.
- Speed: DiCE takes 70–1,043× as long as PACE.
- Points per seed where both are valid: 27–30 of 30.
