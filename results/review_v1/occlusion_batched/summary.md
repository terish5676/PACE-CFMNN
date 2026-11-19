# PACE-flip against batched occlusion

Same models, test points and background as the paper's timing (`run_faithfulness.py`; 50 test points per seed, 36 for Wine; seeds [0, 1, 2]; 32 values per feature). Milliseconds per explained point, mean over seeds. PACE-flip and batched occlusion: median of 3 runs; point-by-point occlusion: one run, as in the paper's timing table. Times depend on the machine; compare within a row. A descriptive check added after the pre-submission review, not planned in advance.

| Dataset | PACE-flip | Occlusion, point by point | Occlusion, batched | Point by point vs PACE-flip | Batched vs PACE-flip | Batching gain | Largest difference |
|---|---|---|---|---|---|---|---|
| S1 | 1.06 | 165.2 | 99.2 | 156× | 93× | 1.7× | 1.1e-16 |
| S2 | 1.09 | 119.9 | 84.2 | 110× | 78× | 1.4× | 1.2e-16 |
| S3 | 0.928 | 119.5 | 84.0 | 129× | 91× | 1.4× | 1.2e-16 |
| Wine | 0.110 | 5.64 | 3.53 | 51× | 32× | 1.6× | 1.1e-16 |
| Breast Cancer | 0.272 | 21.1 | 20.2 | 78× | 74× | 1.0× | 6.1e-16 |
| Thyroid | 0.222 | 14.8 | 11.4 | 67× | 51× | 1.3× | 4.4e-15 |
| Shuttle | 0.046 | 3.55 | 1.59 | 77× | 34× | 2.2× | 2.1e-15 |
| MAGIC | 0.692 | 19.5 | 19.2 | 28× | 28× | 1.0× | 3.3e-15 |
| Letter | 1.49 | 108.6 | 100.9 | 73× | 68× | 1.1× | 2.2e-16 |
| Landsat | 0.944 | 127.4 | 126.8 | 135× | 134× | 1.0× | 6.1e-16 |
| Page Blocks | 0.136 | 4.15 | 3.37 | 31× | 25× | 1.2× | 5.4e-15 |

- **PACE-flip vs batched occlusion:** 25–134× faster.
- **PACE-flip vs point-by-point occlusion (this run):** 28–156× faster.
- **Batching** makes occlusion 1.0–2.2× faster.
- **Same scores:** batched occlusion differs from point-by-point occlusion and from PACE-flip by at most 5.4e-15 (floating-point rounding).
