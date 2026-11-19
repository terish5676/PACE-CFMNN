# Run P4: exact PACE-SHAP for GFMM-family networks

Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md). Written by `scripts/run_gfmm_exact.py`.

**Check: FAIL.** The adapter predicts like hyperbox-brain on every explained point; PACE-SHAP-min equals the brute force within 1e-12 on every point; where timed, shap's ExactExplainer agrees within 1e-09.
- S2 seed 310 IOL-GFMM: the adapter predicts unlike hyperbox-brain
- S2 seed 312 IOL-GFMM: the adapter predicts unlike hyperbox-brain
- S2 seed 314 IOL-GFMM: the adapter predicts unlike hyperbox-brain
- S2 seed 316 IOL-GFMM: the adapter predicts unlike hyperbox-brain
- S2 seed 317 IOL-GFMM: the adapter predicts unlike hyperbox-brain
- S2 seed 318 IOL-GFMM: the adapter predicts unlike hyperbox-brain
- S3 seed 311 IOL-GFMM: the adapter predicts unlike hyperbox-brain
- S3 seed 316 IOL-GFMM: the adapter predicts unlike hyperbox-brain
- S3 seed 318 IOL-GFMM: the adapter predicts unlike hyperbox-brain
- page_blocks seed 311 IOL-GFMM: the adapter predicts unlike hyperbox-brain

## Check (seeds 310–319)

| Dataset | Model | Points explained | Points compared | Boxes (mean) | Largest difference from brute force | (Row, box) pairs skipped |
|---|---|---|---|---|---|---|
| S1 | Online GFMM | 100 | 30 | 2363 | 2.2e-16 | 0.009 |
| S1 | IOL-GFMM | 100 | 30 | 2363 | 2.2e-16 | 0.009 |
| S2 | Online GFMM | 100 | 30 | 2426 | 2.1e-16 | 0.009 |
| S2 | IOL-GFMM | 100 | 30 | 2427 | 2.1e-16 | 0.009 |
| S3 | Online GFMM | 100 | 30 | 2414 | 2.7e-16 | 0.010 |
| S3 | IOL-GFMM | 100 | 30 | 2414 | 2.7e-16 | 0.010 |
| shuttle | Online GFMM | 100 | 30 | 33 | 5.0e-16 | 0.816 |
| shuttle | IOL-GFMM | 100 | 30 | 1129 | 6.7e-16 | 0.739 |
| magic | Online GFMM | 100 | 30 | 323 | 3.6e-16 | 0.703 |
| magic | IOL-GFMM | 100 | 30 | 2255 | 6.7e-16 | 0.673 |
| page_blocks | Online GFMM | 100 | 30 | 84 | 6.8e-16 | 0.712 |
| page_blocks | IOL-GFMM | 100 | 30 | 186 | 3.9e-16 | 0.691 |
