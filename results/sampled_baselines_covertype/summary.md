# Run N: Covertype in the sampled-baseline comparison

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Descriptive: no comparison is tested. Seeds 230–234 (5), 54 features, 50 explained test points per seed, 200 explainer rows, budget 352,000 model rows per explanation. Mean ± standard deviation over seeds. Written by `scripts/run_sampled_baselines_covertype.py` with Run J's task, unchanged.

| Method | Deletion margin AUC (lower is more faithful) | Insertion margin AUC (higher) | Top-3 stability | Model rows per explanation | Milliseconds per explanation, during the run |
|---|---|---|---|---|---|
| PACE-SHAP | -0.2151 ± 0.0163 | 0.0568 ± 0.0264 | 0.980 ± 0.020 | 352,000 ± 0 | 979 ± 37 |
| PermutationSHAP | -0.2153 ± 0.0171 | 0.0560 ± 0.0265 | 0.944 ± 0.017 | 114,777 ± 2,888 | 2,340 ± 118 |
| KernelSHAP-paired | -0.2128 ± 0.0171 | 0.0514 ± 0.0248 | 0.978 ± 0.013 | 351,801 ± 0 | 7,040 ± 314 |
| Random | -0.0930 ± 0.0116 | -0.1042 ± 0.0241 | 0.034 ± 0.016 | 0 ± 0 | 0 ± 0 |

Per seed, PACE-SHAP minus each baseline, deletion margin AUC (negative favours PACE-SHAP; not tested):

| Seed | PermutationSHAP | KernelSHAP-paired |
|---|---|---|
| 230 | +0.0024 | +0.0002 |
| 231 | +0.0006 | +0.0006 |
| 232 | +0.0014 | -0.0028 |
| 233 | -0.0034 | -0.0067 |
| 234 | +0.0001 | -0.0030 |

Every baseline within the budget: yes.

## Milliseconds per explanation, measured alone (seeds [230, 231, 232], 10 points each)

| Method | Milliseconds | Times PACE-SHAP's |
|---|---|---|
| PACE-SHAP | 434 | 1.0× |
| PermutationSHAP | 1,991 | 4.6× |
| KernelSHAP-paired | 6,029 | 13.9× |

